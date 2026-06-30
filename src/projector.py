"""
projector.py - Configurable output projection layer.

Takes a canonical CandidateProfile and a runtime config dict,
and returns a plain dict shaped according to that config.

Config structure:
  {
    "fields": [
      { "path": "full_name", "type": "string", "required": true },
      { "path": "primary_email", "from": "emails[0]", "type": "string", "required": false },
      { "path": "phone", "from": "phones[0]", "type": "string", "normalize": "E164" },
      { "path": "skill_names", "from": "skills[].name", "type": "string[]" }
    ],
    "include_confidence": true,
    "include_provenance": false,
    "on_missing": "null"   // "null" | "omit" | "error"
  }

If no config is given, the full canonical schema is emitted.
"""

from __future__ import annotations
import dataclasses
import logging
import re
from typing import Any, Optional

from schema import CandidateProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialise canonical profile to plain dict
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    return obj


def profile_to_full_dict(profile: CandidateProfile) -> dict:
    """Return the complete canonical profile as a plain dict."""
    d = _to_dict(profile)
    # Rename field_of_study back to field for the public schema
    for edu in d.get("education", []):
        if "field_of_study" in edu:
            edu["field"] = edu.pop("field_of_study")
    return d


# ---------------------------------------------------------------------------
# Path accessor helpers
# ---------------------------------------------------------------------------

def _get_path(data: dict, path: str) -> Any:
    """
    Access a nested path with dot notation and array operators.

    Supported patterns:
      "full_name"            → data["full_name"]
      "location.city"        → data["location"]["city"]
      "emails[0]"            → data["emails"][0]
      "skills[].name"        → [s["name"] for s in data["skills"]]
    """
    # Handle array spread: "skills[].name"
    spread_match = re.match(r"^(\w+)\[\]\.(\w+)$", path)
    if spread_match:
        key, sub = spread_match.group(1), spread_match.group(2)
        items = data.get(key) or []
        if not isinstance(items, list):
            return None
        return [item.get(sub) for item in items if isinstance(item, dict) and sub in item]

    # Handle index access, optionally followed by further dotted path:
    # "emails[0]"  or  "experience[0].title"  or  "experience[0].company"
    index_match = re.match(r"^(\w+)\[(\d+)\](?:\.(.+))?$", path)
    if index_match:
        key, idx, rest = index_match.group(1), int(index_match.group(2)), index_match.group(3)
        items = data.get(key)
        if not isinstance(items, list) or len(items) <= idx:
            return None
        value = items[idx]
        if rest is None:
            return value
        # Recurse on the remaining dotted path against the indexed element
        return _get_path(value, rest) if isinstance(value, dict) else None

    # Handle nested dot: "location.city"
    parts = path.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


# ---------------------------------------------------------------------------
# Type coercions
# ---------------------------------------------------------------------------

def _coerce(value: Any, type_hint: str | None) -> Any:
    if value is None:
        return None
    if not type_hint:
        return value

    t = type_hint.lower().strip()
    if t == "string":
        return str(value) if value is not None else None
    if t == "string[]":
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        return [str(value)]
    if t == "number":
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    if t == "boolean":
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Normalization at projection time
# ---------------------------------------------------------------------------

def _apply_normalize(value: Any, normalize: str | None) -> Any:
    if not normalize or value is None:
        return value

    norm = normalize.lower()

    if norm == "e164":
        from normalizers import normalize_phone
        if isinstance(value, str):
            e164, ok = normalize_phone(value)
            return e164 if ok else value
        return value

    if norm == "canonical":
        from normalizers import canonicalize_skill
        if isinstance(value, str):
            canonical, _ = canonicalize_skill(value)
            return canonical
        if isinstance(value, list):
            result = []
            for v in value:
                canonical, _ = canonicalize_skill(str(v))
                result.append(canonical or v)
            return result
        return value

    if norm in ("lowercase", "lower"):
        if isinstance(value, str):
            return value.lower()
        return value

    if norm in ("uppercase", "upper"):
        if isinstance(value, str):
            return value.upper()
        return value

    return value


# ---------------------------------------------------------------------------
# Projection entry point
# ---------------------------------------------------------------------------

class ProjectionError(Exception):
    """Raised when a required field is missing and on_missing='error'."""


def project(profile: CandidateProfile, config: dict | None) -> dict:
    """
    Apply the runtime config to a canonical profile and return the projected dict.

    If config is None, returns the full canonical dict.
    """
    if not config:
        return profile_to_full_dict(profile)

    full = profile_to_full_dict(profile)
    fields_config = config.get("fields")
    include_confidence = config.get("include_confidence", True)
    include_provenance = config.get("include_provenance", False)
    on_missing = config.get("on_missing", "null")  # "null" | "omit" | "error"

    if not fields_config:
        # No field config → full output, honour flags
        result = {k: v for k, v in full.items()
                  if k not in ("overall_confidence", "provenance")}
        if include_confidence:
            result["overall_confidence"] = full.get("overall_confidence")
        if include_provenance:
            result["provenance"] = full.get("provenance", [])
        return result

    result: dict[str, Any] = {}

    for field_spec in fields_config:
        out_key = field_spec.get("path")          # output key name
        src_path = field_spec.get("from", out_key)  # canonical path to read from
        type_hint = field_spec.get("type")
        required = field_spec.get("required", False)
        normalize = field_spec.get("normalize")

        if not out_key:
            logger.warning("projector: field spec missing 'path', skipping: %s", field_spec)
            continue

        value = _get_path(full, src_path)
        value = _apply_normalize(value, normalize)
        value = _coerce(value, type_hint)

        if value is None:
            if required:
                if on_missing == "error":
                    raise ProjectionError(
                        f"Required field '{out_key}' (from '{src_path}') is missing in profile {profile.candidate_id}"
                    )
                elif on_missing == "omit":
                    continue
                else:  # "null" or anything else
                    result[out_key] = None
            else:
                if on_missing == "omit":
                    continue
                result[out_key] = None
        else:
            result[out_key] = value

    if include_confidence:
        result["overall_confidence"] = full.get("overall_confidence")

    if include_provenance:
        result["provenance"] = full.get("provenance", [])

    return result


# ---------------------------------------------------------------------------
# Schema validation (lightweight)
# ---------------------------------------------------------------------------

def validate_output(projected: dict, config: dict | None) -> list[str]:
    """
    Validate a projected dict against the field config.
    Returns a list of validation error strings (empty = valid).
    """
    errors: list[str] = []
    if not config or not config.get("fields"):
        return errors

    for field_spec in config.get("fields", []):
        out_key = field_spec.get("path")
        required = field_spec.get("required", False)
        type_hint = field_spec.get("type", "").lower()

        if not out_key:
            continue

        value = projected.get(out_key)

        if required and value is None:
            errors.append(f"REQUIRED field '{out_key}' is null in output")
            continue

        if value is not None and type_hint:
            if type_hint == "string" and not isinstance(value, str):
                errors.append(f"Field '{out_key}' expected string, got {type(value).__name__}")
            elif type_hint == "string[]":
                if not isinstance(value, list):
                    errors.append(f"Field '{out_key}' expected string[], got {type(value).__name__}")
                elif not all(isinstance(v, str) for v in value):
                    errors.append(f"Field '{out_key}': not all items are strings")
            elif type_hint == "number" and not isinstance(value, (int, float)):
                errors.append(f"Field '{out_key}' expected number, got {type(value).__name__}")

    return errors
