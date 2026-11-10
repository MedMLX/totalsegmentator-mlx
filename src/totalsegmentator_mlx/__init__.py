"""TotalSegmentator v2 ``total`` CT inference on Apple Silicon (MLX), research software.

``segment`` runs a CT through the native workflow and returns uint8 global label IDs
(1-117) on the input grid. The function loads on first access, so importing the
package does not import the MLX engine.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from totalsegmentator_mlx.engine import (
        clear_predictor_cache,
        configure_predictor_cache,
        predictor_cache_info,
    )
    from totalsegmentator_mlx.pipeline import segment

__all__ = ["clear_predictor_cache", "configure_predictor_cache", "predictor_cache_info", "segment"]


def __getattr__(name: str) -> object:
    """Load the segmentation entry point lazily on first access."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    owner = "pipeline" if name == "segment" else "engine"
    return getattr(import_module(f"{__name__}.{owner}"), name)
