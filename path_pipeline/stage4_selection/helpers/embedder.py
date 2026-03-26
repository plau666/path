"""Embed tables into fixed-length vectors using EmbeddingGemma.

Each table is embedded as concat(schema_embedding, mean_row_embedding):
  1. Embed the schema header ("Columns: charttime, heartrate, ...").
  2. Embed each serialized row individually, then average.
  3. Concatenate [schema_emb, mean_row_emb] to form the final vector.

This captures both the column structure and the average row content,
following the PATH paper's approach of combining schema and row embeddings.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from path_pipeline.stage1_preprocess.helpers.data_loading import FEATURE_COLUMNS
from path_pipeline.stage1_preprocess.helpers.serialization import (
    serialize_row,
    serialize_schema,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "google/embeddinggemma-300m"
PROMPT_TEMPLATE = "task: sentence similarity | query: {content}"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME, device: str = "cuda"):
    """Load the HuggingFace embedding model (tokenizer + model).

    Args:
        model_name: HuggingFace model ID.
        device: Device to load the model on.

    Returns:
        Dict with 'tokenizer', 'model', and 'device'.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded embedding model: {model_name} on {device}")
    return {"tokenizer": tokenizer, "model": model, "device": device}


def _encode(embedding_model, texts: List[str], batch_size: int) -> np.ndarray:
    """Encode texts with the prompt template prepended, using mean pooling.

    Args:
        embedding_model: Dict with 'tokenizer', 'model', 'device'.
        texts: List of strings to encode.
        batch_size: Batch size for encoding.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim).
    """
    tokenizer = embedding_model["tokenizer"]
    model = embedding_model["model"]
    device = embedding_model["device"]

    prompted = [PROMPT_TEMPLATE.format(content=t) for t in texts]

    all_embeddings = []
    for i in range(0, len(prompted), batch_size):
        batch = prompted[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        # Mean pooling over non-padding tokens
        attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        token_embs = outputs.last_hidden_state  # (B, T, D)
        summed = (token_embs * attention_mask).sum(dim=1)  # (B, D)
        counts = attention_mask.sum(dim=1)  # (B, 1)
        mean_pooled = summed / counts  # (B, D)

        all_embeddings.append(mean_pooled.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


def _rows_from_dict_list(
    rows: List[Dict[str, str]], columns: List[str]
) -> List[str]:
    """Serialize a list of row dicts into individual row strings.

    Goes through pd.Series + serialize_row to ensure consistent number
    formatting (e.g. pain rounded to int, NaN handled).

    Args:
        rows: List of dicts mapping column name -> value string.
        columns: Column names for serialization.

    Returns:
        List of serialized row strings.
    """
    result = []
    for i, row_dict in enumerate(rows):
        series = pd.Series(row_dict)
        result.append(serialize_row(series, row_index=i + 1, columns=columns))
    return result


def _parse_table_text(text: str) -> tuple:
    """Split a pre-serialized table text into schema string and row strings.

    Args:
        text: Full serialized table ("Columns: ...\\n[Row 1]: ...\\n[Row 2]: ...")

    Returns:
        (schema_str, list_of_row_strs)
    """
    lines = text.strip().split("\n")
    schema = lines[0]
    rows = [line for line in lines[1:] if line.strip()]
    return schema, rows


def embed_tables(
    model,
    tables: List[List[Dict[str, str]]],
    columns: Optional[List[str]] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed tables as concat(schema_embedding, mean_row_embedding).

    Args:
        model: Embedding model dict from load_embedding_model().
        tables: List of tables, where each table is a list of row dicts.
        columns: Column names for serialization.
        batch_size: Batch size for encoding.

    Returns:
        np.ndarray of shape (n_tables, 2 * embedding_dim).
    """
    if columns is None:
        columns = FEATURE_COLUMNS

    # 1. Schema embedding (same for all tables)
    schema_text = serialize_schema(columns)
    schema_emb = _encode(model, [schema_text], batch_size=batch_size)[0]

    # 2. Collect all row texts across all tables for batch encoding
    all_row_texts = []
    table_row_counts = []
    for table_rows in tables:
        row_texts = _rows_from_dict_list(table_rows, columns)
        all_row_texts.extend(row_texts)
        table_row_counts.append(len(row_texts))

    # 3. Batch-encode all rows at once
    if all_row_texts:
        all_row_embs = _encode(model, all_row_texts, batch_size=batch_size)
    else:
        all_row_embs = np.zeros((0, schema_emb.shape[0]))

    # 4. Compute mean row embedding per table, then concat with schema
    embeddings = []
    offset = 0
    for count in table_row_counts:
        if count > 0:
            mean_row_emb = all_row_embs[offset : offset + count].mean(axis=0)
        else:
            mean_row_emb = np.zeros_like(schema_emb)
        offset += count
        embeddings.append(np.concatenate([schema_emb, mean_row_emb]))

    return np.array(embeddings)


def embed_texts(
    model,
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Embed pre-serialized table texts as concat(schema_emb, mean_row_emb).

    Each text is expected to be a full serialized table:
        "Columns: ...\\n[Row 1]: ...\\n[Row 2]: ..."

    Args:
        model: Embedding model dict from load_embedding_model().
        texts: List of serialized table strings.
        batch_size: Batch size for encoding.

    Returns:
        np.ndarray of shape (n_texts, 2 * embedding_dim).
    """
    # Parse each table text into schema + rows
    schemas = []
    all_row_texts = []
    table_row_counts = []
    for text in texts:
        schema, rows = _parse_table_text(text)
        schemas.append(schema)
        all_row_texts.extend(rows)
        table_row_counts.append(len(rows))

    # Encode unique schemas (typically all identical) and all rows in batch
    unique_schemas = list(dict.fromkeys(schemas))
    schema_embs = _encode(model, unique_schemas, batch_size=batch_size)
    schema_map = {s: emb for s, emb in zip(unique_schemas, schema_embs)}

    if all_row_texts:
        all_row_embs = _encode(model, all_row_texts, batch_size=batch_size)
    else:
        dim = schema_embs.shape[1]
        all_row_embs = np.zeros((0, dim))

    # Build per-table embeddings: concat(schema_emb, mean_row_emb)
    embeddings = []
    offset = 0
    for schema, count in zip(schemas, table_row_counts):
        schema_emb = schema_map[schema]
        if count > 0:
            mean_row_emb = all_row_embs[offset : offset + count].mean(axis=0)
        else:
            mean_row_emb = np.zeros_like(schema_emb)
        offset += count
        embeddings.append(np.concatenate([schema_emb, mean_row_emb]))

    return np.array(embeddings)
