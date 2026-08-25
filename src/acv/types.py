"""Core data schemas and strict enum classifications.

Defines the Pydantic models for the entire extraction and audit pipeline.
Ensures strict serialization and validation of LLM outputs and metric aggregations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

SCHEMA_VERSION = "v1"

# =============================================================================
#                    ********* CORPUS DEFINITIONS *********                  
#         Metadata and classification for unstructured manuscript artifacts.   
# =============================================================================

class Family(str, Enum):
    UNITARY = "unitary"
    BINARY = "binary"
    TERNARY = "ternary"
    QUATERNARY = "quaternary"
    UNKNOWN = "unknown"


class Paper(BaseModel):
    openalex_id: str
    doi: Optional[str] = None
    title: str = ""
    year: Optional[int] = None
    venue: Optional[str] = None
    is_oa: bool = False
    oa_url: Optional[str] = None
    oa_locations: list[str] = Field(default_factory=list)
    arxiv_id: Optional[str] = None
    matched_terms: list[str] = Field(default_factory=list)
    cited_by_count: int = 0
    formula: Optional[str] = None
    family: Family = Family.UNKNOWN

    @property
    def retrievable(self) -> bool:
        return bool(self.arxiv_id or self.oa_url)

    @property
    def key(self) -> str:
        base = self.doi or self.openalex_id
        return base.replace("/", "_").replace(":", "_").lstrip("_")

# =============================================================================
#                    ********* EXTRACTION SCHEMAS *********                  
#        Strictly typed targets for large language model data extraction.      
# =============================================================================

class Code(str, Enum):
    VASP = "vasp"
    QUANTUM_ESPRESSO = "quantum_espresso"
    SIESTA = "siesta"
    CASTEP = "castep"
    ABINIT = "abinit"
    CP2K = "cp2k"
    GAUSSIAN = "gaussian"
    OTHER = "other"
    NOT_STATED = "not_stated"


class ElasticUnits(str, Enum):
    N_PER_M = "N/m"
    GPA = "GPa"
    BOTH = "both"
    NOT_STATED = "not_stated"


T = TypeVar("T")


class Reported(BaseModel, Generic[T]):
    reported: bool = False
    value: Optional[T] = None
    evidence: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MethodParameters(BaseModel):
    code: Reported[Code] = Field(default_factory=Reported)
    code_version: Reported[str] = Field(default_factory=Reported)
    xc_functional: Reported[str] = Field(default_factory=Reported)
    pseudopotential_type: Reported[str] = Field(default_factory=Reported)
    k_mesh: Reported[str] = Field(default_factory=Reported)
    smearing: Reported[str] = Field(default_factory=Reported)
    force_threshold_ev_ang: Reported[float] = Field(default_factory=Reported)
    energy_threshold_ev: Reported[float] = Field(default_factory=Reported)
    spin_polarised: Reported[bool] = Field(default_factory=Reported)
    dispersion_correction: Reported[str] = Field(default_factory=Reported)

    plane_wave_cutoff_ev: Reported[float] = Field(default_factory=Reported)
    augmentation_cutoff_ev: Reported[float] = Field(default_factory=Reported)

    mesh_cutoff_ry: Reported[float] = Field(default_factory=Reported)
    basis_size: Reported[str] = Field(default_factory=Reported)
    pao_energy_shift_ev: Reported[float] = Field(default_factory=Reported)
    basis_split_norm: Reported[float] = Field(default_factory=Reported)

    vacuum_spacing_ang: Reported[float] = Field(default_factory=Reported)
    dipole_correction: Reported[bool] = Field(default_factory=Reported)
    thickness_for_gpa_conversion_ang: Reported[float] = Field(default_factory=Reported)
    elastic_units_reported: Reported[ElasticUnits] = Field(default_factory=Reported)


class Claim(BaseModel):
    property: str
    value: Optional[float] = None
    unit: Optional[str] = None
    material_formula: Optional[str] = None
    structure_prototype: Optional[str] = None
    evidence: Optional[str] = None


class ExtractionStatus(str, Enum):
    OK = "ok"
    NOT_COMPUTATIONAL = "not_computational"
    NO_FULLTEXT = "no_fulltext"
    FAILED = "failed"


class Extraction(BaseModel):
    paper_key: str
    doi: Optional[str] = None
    status: ExtractionStatus = ExtractionStatus.OK
    reasoning: Optional[str] = Field(default=None)
    is_computational: bool = False
    is_pentagonal_2d: bool = False
    method: MethodParameters = Field(default_factory=MethodParameters)
    claims: list[Claim] = Field(default_factory=list)
    error: Optional[str] = None

    schema_version: str = SCHEMA_VERSION
    model: Optional[str] = None
    prompt_id: Optional[str] = None

# =============================================================================
#                     ********* AUDIT AGGREGATES *********                   
#          Reportability scores and deterministic execution manifests.         
# =============================================================================

class ReportabilityScore(BaseModel):
    paper_key: str
    doi: Optional[str] = None
    code: Optional[str] = None
    fields_required: int = 0
    fields_reported: int = 0
    missing: list[str] = Field(default_factory=list)
    reproducible_in_principle: bool = False

    @property
    def fraction_reported(self) -> float:
        return self.fields_reported / self.fields_required if self.fields_required else 0.0


class RunManifest(BaseModel):
    run_id: str
    started_at: str
    stage: str
    model: Optional[str] = None
    prompt_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    n_inputs: int = 0
    n_ok: int = 0
    n_failed: int = 0
    cost_usd: float = 0.0
    params: dict[str, Any] = Field(default_factory=dict)
