"""
schema.py - Canonical candidate profile schema and data models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Location:
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None   # ISO-3166 alpha-2


@dataclass
class Links:
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    other: list[str] = field(default_factory=list)


@dataclass
class Skill:
    name: str
    confidence: float          # 0.0 – 1.0
    sources: list[str] = field(default_factory=list)


@dataclass
class Experience:
    company: Optional[str] = None
    title: Optional[str] = None
    start: Optional[str] = None    # YYYY-MM
    end: Optional[str] = None      # YYYY-MM or None = current
    summary: Optional[str] = None


@dataclass
class Education:
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    end_year: Optional[int] = None


@dataclass
class Provenance:
    field: str
    source: str   # e.g. "recruiter_csv", "ats_json", "github", "notes"
    method: str   # e.g. "direct", "regex", "inferred", "api"


@dataclass
class CandidateProfile:
    candidate_id: str
    full_name: Optional[str] = None
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)           # E.164
    location: Optional[Location] = None
    links: Optional[Links] = None
    headline: Optional[str] = None
    years_experience: Optional[float] = None
    skills: list[Skill] = field(default_factory=list)
    experience: list[Experience] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)
    overall_confidence: float = 0.0
