"""TotalSegmentator v2 ``total`` CT inference on Apple Silicon (MLX), research software.

``segment`` runs a CT through the native workflow and writes uint8 global label IDs
(1-117) on the input grid; ``load_model`` keeps the networks loaded across CTs. Both
load on first access, so importing the package does not import the MLX engine.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from totalsegmentator_mlx.medmlx import load_model, segment

__all__ = ["load_model", "segment"]


def __getattr__(name: str) -> object:
    """Load the public operation lazily on first access."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.medmlx"), name)
