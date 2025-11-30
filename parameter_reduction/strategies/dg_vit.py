# dg_vit.py
"""
Depth-Grouped ViT (DG-ViT) built on top of timm's VisionTransformer.

Core idea:
    - Start from the canonical ViT-B/16 (`vit_base_patch16_224`)
      with its usual initialization and configuration.
    - After the model is fully built (and optionally pretrained),
      tie MLP sub-blocks between adjacent transformer blocks:
          (0, 1), (2, 3), ..., (10, 11)
      using GroupedMLPStrategy.
    - Optionally rescale the shared MLP parameters by 1/sqrt(2) so that
      gradient variance stays matched when two depth positions share one MLP.

This gives the exact "born 86M → 58M" story:
    - Initialization happens in the full 86M parameter space.
    - Only then do we impose the parameter sharing structure.
    - Data config, patch embedding, attention, norms, etc. are
      identical to the base ViT-B/16 implementation in timm.
"""

import logging
import math

import torch.distributed as dist
from torch.nn import Module

from timm.models import vision_transformer as vit
from timm.models._registry import register_model

from parameter_reduction import GroupedMLPStrategy  # your existing implementation

_logger = logging.getLogger(__name__)

# Default scale = 1/sqrt(2): variance-preserving for 2-way sharing
_DEFAULT_DG_SCALE = 1.0 / math.sqrt(2.0)


def is_primary() -> bool:
    """Return True if this process should handle 'global' logging."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def apply_depth_grouped_mlp(
    model: Module,
    scale_factor: float = _DEFAULT_DG_SCALE,
) -> Module:
    """
    In-place convert a ViT-B/16 into a Depth-Grouped MLP variant.

    Assumptions:
        - `model` is a timm VisionTransformer with 12 blocks
          (ViT-B/16 style) and each block has an `.mlp` submodule.
        - Grouping pattern is fixed to adjacent pairs:
              (0, 1), (2, 3), ..., (10, 11)
          as implemented in GroupedMLPStrategy.

    Transformation (as in GroupedMLPStrategy):
        For each pair (a, b):
            - Multiply fc1 / fc2 weights in block a by `scale_factor`.
            - Set block b's `.mlp` to be the same module as block a's `.mlp`.
            - fc2.bias is NOT scaled.

    This preserves:
        - The original initialization in the full 86M parameter space.
        - All non-MLP parameters (attn, norms, embeddings, etc.).
        - All timm default configs / data configs attached to the model.

    Returns:
        The same model instance, modified in-place.
    """
    strategy = GroupedMLPStrategy(scale_factor=scale_factor)
    model = strategy.apply(model)

    # Attach introspection metadata (useful for logging / sanity checks)
    model.dg_mlp_scale = float(scale_factor)
    model.dg_mlp_grouping = strategy.grouping
    model.dg_mlp_unique_mlps = len(strategy.unique_mlps)

    # Log stats only on primary process to avoid duplicate output in DDP
    if is_primary():
        try:
            strategy.log_stats(model, _logger)
        except Exception:
            # Never let logging failure break model creation
            pass

    return model


@register_model
def vit_base_patch16_224_dgmlp(
    pretrained: bool = False,
    mlp_scale: float = _DEFAULT_DG_SCALE,
    **kwargs,
) -> vit.VisionTransformer:
    """
    Depth-Grouped MLP ViT-B/16 @ 224 ("DG-ViT-B/16").

    Construction is deliberately two-stage:

        1) Build a completely standard ViT-B/16 from timm:
               base = vit.vit_base_patch16_224(pretrained=pretrained, **kwargs)

        2) Apply depth-grouped MLP tying post-init:
               dg = apply_depth_grouped_mlp(base, scale_factor=mlp_scale)

    This guarantees:
        - Same architecture as timm's ViT-B/16 (same depth, dim, heads, etc.).
        - Same weight initialization and `fix_init` behaviour.
        - Same `pretrained_cfg` / `default_cfg` and thus identical
          data pipeline (input size, mean/std, crop_pct, etc.).
        - The only difference is the MLP parameter sharing.

    Args:
        pretrained:
            Passed directly to `vit.vit_base_patch16_224`.
            If True, we load the usual ViT-B/16 weights and then
            tie MLPs; if False, we train DG-ViT from scratch.
        mlp_scale:
            Scaling factor for the shared MLP weights:
              - 1.0: no rescaling → shared MLP sees effectively
                doubled gradient magnitude from two depth positions.
              - 1 / sqrt(2): (default) variance-preserving scaling
                for 2-way sharing.
        **kwargs:
            Any other timm-compatible VisionTransformer kwargs
            (img_size, num_classes, drop_path, etc.) are passed
            unchanged into `vit.vit_base_patch16_224`.

    Returns:
        A VisionTransformer instance whose MLPs are depth-grouped and shared.
    """
    # 1. Build canonical ViT-B/16 from the timm implementation
    base_model = vit.vit_base_patch16_224(pretrained=pretrained, **kwargs)

    # 2. Apply grouped-MLP parameter sharing post-init
    dg_model = apply_depth_grouped_mlp(base_model, scale_factor=mlp_scale)

    return dg_model
