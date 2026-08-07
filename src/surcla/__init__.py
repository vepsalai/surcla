"""SurCla: surrogate model recommendation for regression datasets."""

__version__ = "1.0.0.dev0"

from .decoder import RegretDecoder
from .metafeatures import SCHEMA_VERSION, manual_metafeature_vector
from .recommend import Candidate, Report, recommend

__all__ = ["recommend", "Report", "Candidate", "RegretDecoder",
           "manual_metafeature_vector", "SCHEMA_VERSION", "__version__"]
