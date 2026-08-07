"""SurCla: surrogate model recommendation for regression datasets."""

__version__ = "1.0.0.dev0"

from .decoder import RegretDecoder
from .metafeatures import SCHEMA_VERSION, manual_metafeature_vector

__all__ = ["RegretDecoder", "manual_metafeature_vector", "SCHEMA_VERSION",
           "__version__"]
