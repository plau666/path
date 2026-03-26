"""Differentially private nearest-neighbor voting for synthetic table selection.

Implements the private selection mechanism from the PATH paper (Section 5),
adapted from the DP_NN_HISTOGRAM procedure in Aug-PE (Rosenblatt et al., 2024):

1. Embed all real and synthetic tables into a shared vector space.
2. Each real table votes for its k nearest synthetic neighbors.
3. Gaussian noise is added to the vote histogram (sensitivity = k).
4. The top-N synthetic tables by noisy vote count form the final dataset.

Privacy guarantee: The voting step satisfies (epsilon_select, delta)-DP via
the Gaussian mechanism. Each real table contributes exactly k votes, so the
L2 sensitivity of the histogram is sqrt(k) (each table changes at most k bins
by 1 each, so L2 sensitivity = sqrt(k)).
"""

import logging
from collections import Counter
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _nn_search_faiss(syn_embedding, priv_embedding, k):
    """Find k nearest synthetic neighbors for each private embedding using FAISS.

    Uses cosine similarity (inner product on L2-normalized vectors).
    Automatically uses all available GPUs.

    Args:
        syn_embedding: (n_syn, dim) float32 array.
        priv_embedding: (n_real, dim) float32 array.
        k: Number of nearest neighbors.

    Returns:
        ids: (n_real, k) array of nearest-neighbor indices into syn_embedding.
    """
    import faiss

    syn_embedding = np.ascontiguousarray(syn_embedding, dtype=np.float32)
    priv_embedding = np.ascontiguousarray(priv_embedding, dtype=np.float32)

    # L2-normalize for cosine similarity via inner product
    faiss.normalize_L2(syn_embedding)
    faiss.normalize_L2(priv_embedding)

    index = faiss.IndexFlatIP(syn_embedding.shape[1])

    ngpus = faiss.get_num_gpus()
    if ngpus > 0:
        co = faiss.GpuMultipleClonerOptions()
        index = faiss.index_cpu_to_all_gpus(index, co, ngpus)
        logger.info(f"FAISS using {ngpus} GPU(s)")
    else:
        logger.info("FAISS using CPU")

    index.add(syn_embedding)
    _, ids = index.search(priv_embedding, k)
    return ids


def dp_nn_histogram(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    k: int = 10,
    sigma: Optional[float] = None,
    epsilon: Optional[float] = None,
    delta: float = 1e-10,
    seed: int = 42,
) -> np.ndarray:
    """Compute a differentially private nearest-neighbor vote histogram.

    Each real table votes for its k nearest synthetic neighbors. Gaussian noise
    is added to satisfy (epsilon, delta)-DP.

    The L2 sensitivity is sqrt(k): adding or removing one real table changes at
    most k histogram bins by 1 each, giving L2 norm = sqrt(k).

    Noise calibration (Gaussian mechanism):
        sigma = sqrt(k) * sqrt(2 * ln(1.25 / delta)) / epsilon

    Args:
        real_embeddings: (n_real, dim) array of real table embeddings.
        synthetic_embeddings: (n_syn, dim) array of synthetic table embeddings.
        k: Number of nearest synthetic neighbors each real table votes for.
        sigma: Noise standard deviation. If provided, overrides epsilon/delta.
        epsilon: Privacy budget for the selection step.
        delta: Privacy parameter delta.
        seed: Random seed for noise generation.

    Returns:
        Noisy vote histogram of shape (n_syn,).
    """
    n_real = real_embeddings.shape[0]
    n_syn = synthetic_embeddings.shape[0]

    if sigma is None:
        if epsilon is None:
            raise ValueError("Either sigma or epsilon must be provided")
        # Gaussian mechanism: sigma = L2_sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        l2_sensitivity = np.sqrt(k)
        sigma = l2_sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        logger.info(
            f"Calibrated sigma={sigma:.4f} for epsilon={epsilon}, delta={delta:.2e}, "
            f"k={k}, L2_sensitivity={l2_sensitivity:.4f}"
        )

    # Find k nearest synthetic neighbors for each real table via FAISS
    ids = _nn_search_faiss(synthetic_embeddings, real_embeddings, k)

    # Count votes
    histogram = np.zeros(n_syn, dtype=np.float64)
    vote_counts = Counter(ids.flatten().tolist())
    histogram[list(vote_counts.keys())] = list(vote_counts.values())

    logger.info(
        f"Voting complete: {n_real} real tables voted for {k} neighbors each. "
        f"Total votes: {histogram.sum():.0f}, "
        f"max votes: {histogram.max():.0f}, "
        f"non-zero bins: {(histogram > 0).sum()}/{n_syn}"
    )

    # Add Gaussian noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=n_syn)
    noisy_histogram = histogram + noise

    logger.info(
        f"Added Gaussian noise: sigma={sigma:.4f}, "
        f"noise L2 norm={np.linalg.norm(noise):.2f}"
    )

    return noisy_histogram


def select_top_n(
    noisy_histogram: np.ndarray,
    n_select: int,
) -> np.ndarray:
    """Select the top-N synthetic tables by noisy vote count.

    Args:
        noisy_histogram: Noisy vote counts of shape (n_syn,).
        n_select: Number of synthetic tables to select.

    Returns:
        Array of indices into the synthetic table list, sorted by descending
        noisy vote count.
    """
    n_select = min(n_select, len(noisy_histogram))
    # argsort descending, take top n_select
    selected = np.argsort(noisy_histogram)[::-1][:n_select]
    logger.info(
        f"Selected top {n_select} from {len(noisy_histogram)} candidates. "
        f"Noisy vote range of selected: "
        f"[{noisy_histogram[selected[-1]]:.1f}, {noisy_histogram[selected[0]]:.1f}]"
    )
    return selected


def private_selection(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    n_select: int,
    k: int = 10,
    epsilon: Optional[float] = None,
    sigma: Optional[float] = None,
    delta: float = 1e-10,
    seed: int = 42,
) -> np.ndarray:
    """End-to-end private selection: vote, add noise, select top-N.

    Args:
        real_embeddings: (n_real, dim) embeddings of real tables.
        synthetic_embeddings: (n_syn, dim) embeddings of synthetic candidates.
        n_select: Number of synthetic tables to include in the final dataset.
        k: Number of nearest neighbors each real table votes for.
        epsilon: Privacy budget for selection (epsilon_select).
        sigma: Direct noise scale (overrides epsilon/delta calibration).
        delta: Privacy parameter delta.
        seed: Random seed.

    Returns:
        Array of selected indices into the synthetic table list.
    """
    noisy_histogram = dp_nn_histogram(
        real_embeddings=real_embeddings,
        synthetic_embeddings=synthetic_embeddings,
        k=k,
        sigma=sigma,
        epsilon=epsilon,
        delta=delta,
        seed=seed,
    )
    selected_indices = select_top_n(noisy_histogram, n_select)
    return selected_indices
