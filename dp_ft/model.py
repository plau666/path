import logging
from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("path")


def load_model(model_name: str, use_dp: bool = True, torch_dtype=torch.bfloat16):
    """Load a HuggingFace causal LM.

    When DP is enabled, uses eager attention (required for Opacus per-sample gradients).
    """
    kwargs = {
        "dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if use_dp:
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    logger.info(f"Loaded model: {model_name} (attn={'eager' if use_dp else 'default'}, dtype={torch_dtype})")
    return model


def load_tokenizer(model_name: str):
    """Load tokenizer and ensure pad token is set."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def apply_lora(
    model,
    r: int = 128,
    lora_alpha: int = 256,
    target_modules: Optional[List[str]] = None,
):
    """Apply LoRA adapters. lora_dropout is forced to 0 for Opacus compatibility."""
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,  # Must be 0 for Opacus per-sample gradient computation
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def prepare_for_opacus(model):
    """Validate and fix model for Opacus compatibility."""
    from opacus.validators import ModuleValidator

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.warning(f"Opacus found {len(errors)} incompatible modules, attempting to fix...")
        model = ModuleValidator.fix(model)
        # Re-validate
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            logger.error(f"Could not fix all modules: {errors}")
            raise RuntimeError(f"Model has {len(errors)} Opacus-incompatible modules after fix attempt")
    logger.info("Model validated for Opacus")
    return model


def build_model(
    model_name: str,
    use_dp: bool = True,
    lora_r: int = 128,
    lora_alpha: int = 256,
    lora_target_modules: Optional[List[str]] = None,
    torch_dtype=torch.bfloat16,
):
    """Full pipeline: load model → apply LoRA → validate for Opacus."""
    model = load_model(model_name, use_dp=use_dp, torch_dtype=torch_dtype)
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules)
    if use_dp:
        model = prepare_for_opacus(model)
    return model
