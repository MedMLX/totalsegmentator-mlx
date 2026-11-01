"""TotalSegmentator v2 ``total`` CT inference on Apple Silicon (MLX), research software.

``segment`` runs a CT through the native workflow and returns uint8 global label IDs
(1-117) on the input grid. The function loads on first access, so importing the
package does not import the MLX engine.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from totalsegmentator_mlx.pipeline import segment

__all__ = ["segment"]


def __getattr__(name: str) -> object:
    """Load the segmentation entry point lazily on first access."""
    if name != "segment":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.pipeline"), name)
