"""
merger.py - Merge engine.

Takes a list of partial CandidateProfile objects (possibly from many sources)
and returns one canonical CandidateProfile with conflicts resolved and
confidence scores assigned.

Merge policy:
  - Identity matching: profiles that share email OR (full_name close match)
    are considered the same person.
  - Field priority: ats_json > recruiter_csv > github_api > recruiter_notes
  - For list fields (emails, phones, skills) we union across sources.
  - For scalar fields we take the highest-priority non-null value.
  - Confidence = f(# sources, field quality, normalized vs raw).
"""

from __future__ import annotations
import re
import logging
from collections import defaultdict
from typing import Optional

from schema import (
    CandidateProfile, Location, Links, Skill, Experience, Education, Provenance
)

logger = logging.getLogger(__name__)

# Source trust priority (higher = more trusted)
_SOURCE_PRIORITY = {
    "ats_json": 4,
    "recruiter_csv": 3,
    "github_api": 2,
    "recruiter_notes": 1,
}


def _source_weight(source_tag: str) -> int:
    for key, weight in _SOURCE_PRIORITY.items():
        if key in source_tag:
            return weight
    return 0


# ---------------------------------------------------------------------------
# Identity matching
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().strip())


def _emails_overlap(a: list[str], b: list[str]) -> bool:
    return bool(set(e.lower() for e in a) & set(e.lower() for e in b))


def _names_match(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return _normalize_name(a) == _normalize_name(b)


def _same_person(p1: CandidateProfile, p2: CandidateProfile) -> bool:
    """Return True if two partial profiles likely represent the same candidate."""
    if _emails_overlap(p1.emails, p2.emails):
        return True
    if _names_match(p1.full_name, p2.full_name):
        return True
    return False


def group_by_identity(profiles: list[CandidateProfile]) -> list[list[CandidateProfile]]:
    """
    Group profiles that represent the same person using union-find.
    Returns list of groups; each group is merged together.
    """
    n = len(profiles)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if _same_person(profiles[i], profiles[j]):
                union(i, j)

    groups: dict[int, list[CandidateProfile]] = defaultdict(list)
    for i, p in enumerate(profiles):
        groups[find(i)].append(p)

    logger.info(
        "merger: grouped %d partial profile(s) into %d identity cluster(s)",
        n, len(groups)
    )
    return list(groups.values())


# ---------------------------------------------------------------------------
# Field-level merge helpers
# ---------------------------------------------------------------------------

def _merge_scalar(values: list[tuple[str, Optional[str]]]) -> Optional[str]:
    """
    Pick the best non-null scalar value from (source_tag, value) pairs.
    Highest-priority source wins.
    """
    best_val = None
    best_pri = -1
    for source_tag, val in values:
        if val is not None:
            pri = _source_weight(source_tag)
            if pri > best_pri:
                best_val = val
                best_pri = pri
    return best_val


def _merge_float(values: list[tuple[str, Optional[float]]]) -> Optional[float]:
    best_val = None
    best_pri = -1
    for source_tag, val in values:
        if val is not None:
            pri = _source_weight(source_tag)
            if pri > best_pri:
                best_val = val
                best_pri = pri
    return best_val


def _merge_list(lists: list[list[str]]) -> list[str]:
    """Union of lists, deduplicated (case-insensitive for strings)."""
    seen: dict[str, str] = {}
    for lst in lists:
        for item in lst:
            key = item.lower()
            if key not in seen:
                seen[key] = item
    return list(seen.values())


def _merge_phones(lists: list[list[str]]) -> list[str]:
    """
    Union of E.164 phone lists, deduplicated by digit body.
    """
    # Map: last-10-digit key → best (longest) E.164 string
    best: dict[str, str] = {}
    for lst in lists:
        for phone in lst:
            digits = phone.lstrip("+")
            key = digits[-10:] if len(digits) >= 10 else digits
            existing = best.get(key)
            if existing is None or len(phone) > len(existing):
                best[key] = phone
    return list(best.values())


def _merge_skills(skill_lists: list[list[Skill]]) -> list[Skill]:
    """
    Union of skills by canonical name.
    If same skill appears in multiple sources, boost confidence and merge source lists.
    """
    merged: dict[str, Skill] = {}
    for skills in skill_lists:
        for s in skills:
            key = s.name.lower()
            if key in merged:
                existing = merged[key]
                # Merge sources
                for src in s.sources:
                    if src not in existing.sources:
                        existing.sources.append(src)
                # Boost confidence with each additional source (capped at 0.95)
                existing.confidence = round(min(0.95, existing.confidence + 0.1), 3)
            else:
                merged[key] = Skill(
                    name=s.name,
                    confidence=s.confidence,
                    sources=list(s.sources),
                )
    return sorted(merged.values(), key=lambda s: -s.confidence)


def _merge_location(locs: list[tuple[str, Optional[Location]]]) -> Optional[Location]:
    """Build best location from available parts."""
    city = _merge_scalar([(src, l.city) for src, l in locs if l])
    region = _merge_scalar([(src, l.region) for src, l in locs if l])
    country = _merge_scalar([(src, l.country) for src, l in locs if l])
    if city or region or country:
        return Location(city=city, region=region, country=country)
    return None


def _merge_links(link_list: list[tuple[str, Optional[Links]]]) -> Optional[Links]:
    linkedin = _merge_scalar([(src, l.linkedin) for src, l in link_list if l])
    github = _merge_scalar([(src, l.github) for src, l in link_list if l])
    portfolio = _merge_scalar([(src, l.portfolio) for src, l in link_list if l])
    other: list[str] = []
    for _, l in link_list:
        if l and l.other:
            other.extend(l.other)
    if linkedin or github or portfolio or other:
        return Links(linkedin=linkedin, github=github, portfolio=portfolio, other=list(set(other)))
    return None


def _merge_experience(exp_lists: list[list[Experience]]) -> list[Experience]:
    """
    Deduplicate experience entries by (company, title).
    Take the most-detailed version of each entry.
    """
    seen: dict[str, Experience] = {}
    for exps in exp_lists:
        for e in exps:
            key = f"{(e.company or '').lower()}|{(e.title or '').lower()}"
            if key not in seen:
                seen[key] = e
            else:
                # Merge: fill in missing fields from duplicate
                existing = seen[key]
                if not existing.start and e.start:
                    existing.start = e.start
                if not existing.end and e.end:
                    existing.end = e.end
                if not existing.summary and e.summary:
                    existing.summary = e.summary
    return list(seen.values())


def _merge_education(edu_lists: list[list[Education]]) -> list[Education]:
    """Deduplicate by institution name."""
    seen: dict[str, Education] = {}
    for edus in edu_lists:
        for e in edus:
            key = (e.institution or "").lower()
            if key not in seen:
                seen[key] = e
    return list(seen.values())


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(profile: CandidateProfile, n_sources: int) -> float:
    """
    Holistic confidence 0–1 based on:
      - How many distinct sources contributed
      - How many core fields are populated
      - Average skill confidence
    """
    score = 0.0

    # Source diversity (up to 0.4)
    score += min(n_sources / 4, 1.0) * 0.4

    # Core field coverage (up to 0.4)
    core_fields = [
        profile.full_name,
        profile.emails[0] if profile.emails else None,
        profile.phones[0] if profile.phones else None,
        profile.location,
        profile.years_experience,
    ]
    filled = sum(1 for f in core_fields if f is not None)
    score += (filled / len(core_fields)) * 0.4

    # Skill confidence average (up to 0.2)
    if profile.skills:
        avg_skill_conf = sum(s.confidence for s in profile.skills) / len(profile.skills)
        score += avg_skill_conf * 0.2

    return round(score, 3)


# ---------------------------------------------------------------------------
# Main merge entry point
# ---------------------------------------------------------------------------

def merge_profiles(group: list[CandidateProfile]) -> CandidateProfile:
    """
    Merge a group of partial profiles (same person, multiple sources)
    into one canonical CandidateProfile.
    """
    if len(group) == 1:
        p = group[0]
        p.overall_confidence = _compute_confidence(p, 1)
        logger.debug("merger: '%s' had a single source, no merge needed", p.full_name)
        return p

    # Collect all provenance
    all_provenance: list[Provenance] = []
    for p in group:
        all_provenance.extend(p.provenance)

    # Which source tags appeared?
    source_tags = list({prov.source for prov in all_provenance})
    n_sources = len(source_tags)

    # Scalar fields: collect (source, value) pairs from provenance + profile
    name_candidates: list[tuple[str, Optional[str]]] = []
    headline_candidates: list[tuple[str, Optional[str]]] = []
    yoe_candidates: list[tuple[str, Optional[float]]] = []
    loc_candidates: list[tuple[str, Optional[Location]]] = []
    links_candidates: list[tuple[str, Optional[Links]]] = []

    all_emails: list[list[str]] = []
    all_phones: list[list[str]] = []
    all_skills: list[list[Skill]] = []
    all_experience: list[list[Experience]] = []
    all_education: list[list[Education]] = []

    for p in group:
        # Determine the dominant source for this partial profile
        prov_sources = [pv.source for pv in p.provenance]
        dominant = max(prov_sources, key=_source_weight) if prov_sources else "unknown"

        name_candidates.append((dominant, p.full_name))
        headline_candidates.append((dominant, p.headline))
        yoe_candidates.append((dominant, p.years_experience))
        loc_candidates.append((dominant, p.location))
        links_candidates.append((dominant, p.links))

        all_emails.append(p.emails)
        all_phones.append(p.phones)
        all_skills.append(p.skills)
        all_experience.append(p.experience)
        all_education.append(p.education)

    full_name = _merge_scalar(name_candidates)
    headline = _merge_scalar(headline_candidates)
    years_experience = _merge_float(yoe_candidates)
    location = _merge_location(loc_candidates)
    links = _merge_links(links_candidates)
    emails = _merge_list(all_emails)
    phones = _merge_phones(all_phones)
    skills = _merge_skills(all_skills)
    experience = _merge_experience(all_experience)
    education = _merge_education(all_education)

    # Use the first found candidate_id (based on best-quality name)
    cid_base = full_name or (emails[0] if emails else "unknown")
    candidate_id = re.sub(r"[^a-z0-9]", "_", (cid_base).lower())

    merged = CandidateProfile(
        candidate_id=candidate_id,
        full_name=full_name,
        emails=emails,
        phones=phones,
        location=location,
        links=links,
        headline=headline,
        years_experience=years_experience,
        skills=skills,
        experience=experience,
        education=education,
        provenance=all_provenance,
    )

    merged.overall_confidence = _compute_confidence(merged, n_sources)
    logger.info(
        "merger: merged '%s' from %d source(s), confidence=%.3f",
        merged.full_name, n_sources, merged.overall_confidence
    )
    return merged
