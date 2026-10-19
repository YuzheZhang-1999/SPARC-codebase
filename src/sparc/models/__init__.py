"""SPARC decoder and spectral teacher components."""

from sparc.models.sparc import SPARC, SPARCInference
from sparc.models.teacher import SpectralTeacher

__all__ = ["SPARC", "SPARCInference", "SpectralTeacher"]
