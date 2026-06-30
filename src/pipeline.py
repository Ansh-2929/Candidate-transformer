"""
pipeline.py - Top-level orchestrator for the candidate data transformer.

Pipeline stages:
  1. Detect    - determine source type from input spec
  2. Extract   - parse each source into partial CandidateProfile objects
  3. Normalize - already applied inside each extractor
  4. Merge     - group by identity, merge conflicts
  5. Confidence- compute overall confidence scores
  6. Project   - apply runtime output config
  7. Validate  - check projected output against schema

Usage (Python API):
  from pipeline import run_pipeline
  result = run_pipeline(sources=[...], config={...})
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any

from schema import CandidateProfile
from extractors import (
    extract_from_recruiter_csv,
    extract_from_ats_json,
    extract_from_github_api,
    extract_from_recruiter_notes,
)
from merger import group_by_identity, merge_profiles
from projector import project, validate_output, ProjectionError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source descriptor types
# ---------------------------------------------------------------------------

def _read_text(path_or_text: str) -> str:
    """Return file contents if path exists, else treat as raw text."""
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return path_or_text


def _extract_source(source_spec: dict) -> list[CandidateProfile]:
    """
    Extract partial profiles from one source specification.

    source_spec keys:
      type: "recruiter_csv" | "ats_json" | "github_api" | "recruiter_notes"
      path: file path  (mutually exclusive with 'data')
      data: raw string (mutually exclusive with 'path')
      name_hint: optional candidate name hint (for github_api)
    """
    src_type = source_spec.get("type", "").lower()
    name_hint = source_spec.get("name_hint", "")

    raw = ""
    if "path" in source_spec:
        try:
            raw = Path(source_spec["path"]).read_text(encoding="utf-8")
        except (OSError, IOError) as exc:
            logger.warning("source '%s': cannot read file '%s': %s", src_type, source_spec["path"], exc)
            return []
    elif "data" in source_spec:
        raw = source_spec["data"]
    else:
        logger.warning("source '%s': neither 'path' nor 'data' provided", src_type)
        return []

    if not raw.strip():
        logger.info("source '%s': empty input, skipping", src_type)
        return []

    profiles: list[CandidateProfile] = []

    if src_type == "recruiter_csv":
        profiles = extract_from_recruiter_csv(raw)
    elif src_type == "ats_json":
        profiles = extract_from_ats_json(raw)
    elif src_type == "github_api":
        p = extract_from_github_api(raw, name_hint)
        if p:
            profiles = [p]
    elif src_type == "recruiter_notes":
        p = extract_from_recruiter_notes(raw)
        if p:
            profiles = [p]
    else:
        logger.warning("Unknown source type '%s', skipping", src_type)

    logger.info("source '%s': extracted %d partial profile(s)", src_type, len(profiles))
    return profiles


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

class PipelineResult:
    def __init__(
        self,
        profiles: list[dict],
        validation_errors: dict[str, list[str]],
        elapsed_ms: float,
        n_sources: int,
        n_merged: int,
    ):
        self.profiles = profiles
        self.validation_errors = validation_errors
        self.elapsed_ms = elapsed_ms
        self.n_sources = n_sources
        self.n_merged = n_merged

    def to_dict(self) -> dict:
        return {
            "meta": {
                "candidate_count": len(self.profiles),
                "sources_processed": self.n_sources,
                "profiles_merged": self.n_merged,
                "elapsed_ms": round(self.elapsed_ms, 1),
                "validation_errors": self.validation_errors,
            },
            "candidates": self.profiles,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    sources: list[dict],
    config: dict | None = None,
    on_missing: str = "null",
) -> PipelineResult:
    """
    Run the full pipeline.

    Args:
        sources:    List of source specification dicts (see _extract_source).
        config:     Optional output projection config.
        on_missing: Fallback on_missing policy if not specified in config.

    Returns:
        PipelineResult with projected candidate dicts and metadata.
    """
    t_start = time.perf_counter()

    # Apply on_missing fallback to config
    if config and "on_missing" not in config:
        config["on_missing"] = on_missing

    # ── Stage 1 & 2: Extract from all sources ──
    all_partials: list[CandidateProfile] = []
    for src in sources:
        partials = _extract_source(src)
        all_partials.extend(partials)

    logger.info("pipeline: %d total partial profiles from %d sources", len(all_partials), len(sources))

    if not all_partials:
        logger.warning("pipeline: no profiles extracted — returning empty result")
        return PipelineResult(
            profiles=[],
            validation_errors={},
            elapsed_ms=(time.perf_counter() - t_start) * 1000,
            n_sources=len(sources),
            n_merged=0,
        )

    # ── Stage 3: Group by identity & merge ──
    groups = group_by_identity(all_partials)
    logger.info("pipeline: %d identity groups from %d partial profiles", len(groups), len(all_partials))

    merged_profiles: list[CandidateProfile] = []
    for group in groups:
        merged = merge_profiles(group)
        merged_profiles.append(merged)

    # ── Stage 4 & 5: Project & validate ──
    projected_list: list[dict] = []
    validation_errors: dict[str, list[str]] = {}

    for profile in merged_profiles:
        try:
            projected = project(profile, config)
        except ProjectionError as exc:
            logger.error("projector: %s", exc)
            # Degrade gracefully: emit null for the failing candidate
            projected = {
                "candidate_id": profile.candidate_id,
                "_projection_error": str(exc),
            }

        errors = validate_output(projected, config)
        if errors:
            validation_errors[profile.candidate_id] = errors

        projected_list.append(projected)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "pipeline: done in %.1f ms — %d candidates, %d with validation errors",
        elapsed_ms, len(projected_list), len(validation_errors)
    )

    return PipelineResult(
        profiles=projected_list,
        validation_errors=validation_errors,
        elapsed_ms=elapsed_ms,
        n_sources=len(sources),
        n_merged=len(merged_profiles),
    )
