"""
extractors.py - Source-specific extraction logic.

Each extractor takes raw input and returns a list of partial CandidateProfile
objects plus provenance records. They never crash on bad input.
"""

from __future__ import annotations
import csv
import json
import re
import logging
from io import StringIO
from typing import Optional

from schema import (
    CandidateProfile, Location, Links, Skill, Experience, Education, Provenance
)
from normalizers import (
    normalize_phone, normalize_date, normalize_country,
    canonicalize_skill, clean_url, validate_email
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prov(field: str, source: str, method: str) -> Provenance:
    return Provenance(field=field, source=source, method=method)


def _id_from_name(name: str) -> str:
    """Generate a stable candidate ID from name (lowercased, no spaces)."""
    return re.sub(r"[^a-z0-9]", "_", (name or "unknown").lower())


def _extract_skills_from_text(text: str, source: str) -> list[Skill]:
    """Pull skills mentioned in free text using a known-skills dictionary."""
    from normalizers import _SKILL_ALIASES
    found: dict[str, Skill] = {}
    text_lower = text.lower()
    for alias, canonical in _SKILL_ALIASES.items():
        # whole-word match to avoid false positives (e.g. "go" in "going")
        if re.search(rf"\b{re.escape(alias)}\b", text_lower):
            if canonical not in found:
                found[canonical] = Skill(name=canonical, confidence=0.6, sources=[source])
    return list(found.values())


def _parse_years_experience(raw) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Structured: Recruiter CSV
# ---------------------------------------------------------------------------

def extract_from_recruiter_csv(csv_text: str) -> list[CandidateProfile]:
    """
    Parse a recruiter CSV export.
    Expected columns: name, email, phone, current_company, title
    Unknown columns are silently ignored; missing columns → None.
    """
    profiles: list[CandidateProfile] = []
    source = "recruiter_csv"

    try:
        reader = csv.DictReader(StringIO(csv_text))
    except Exception as exc:
        logger.warning("recruiter_csv: failed to parse CSV: %s", exc)
        return []

    for i, row in enumerate(reader):
        try:
            name = (row.get("name") or "").strip() or None
            email_raw = (row.get("email") or "").strip() or None
            email, email_ok = validate_email(email_raw)
            if email_raw and not email_ok:
                logger.debug("recruiter_csv row %d: malformed email '%s', dropping", i, email_raw)
            phone_raw = (row.get("phone") or "").strip() or None
            company = (row.get("current_company") or "").strip() or None
            title = (row.get("title") or "").strip() or None

            if not name and not email:
                logger.debug("recruiter_csv row %d: no name or email, skipping", i)
                continue

            cid = _id_from_name(name or email or f"row_{i}")
            prov: list[Provenance] = []

            phone_e164, phone_ok = normalize_phone(phone_raw)

            emails = [email] if email else []
            phones = [phone_e164] if phone_ok else []

            if name:
                prov.append(_prov("full_name", source, "direct"))
            if emails:
                prov.append(_prov("emails", source, "direct"))
            if phones:
                prov.append(_prov("phones", source, "normalized"))
            if not phone_ok and phone_raw:
                logger.debug("recruiter_csv row %d: could not normalize phone '%s'", i, phone_raw)

            experience: list[Experience] = []
            if company or title:
                experience.append(Experience(company=company, title=title))
                prov.append(_prov("experience[0].company", source, "direct"))
                prov.append(_prov("experience[0].title", source, "direct"))

            profile = CandidateProfile(
                candidate_id=cid,
                full_name=name,
                emails=emails,
                phones=phones,
                experience=experience,
                provenance=prov,
            )
            profiles.append(profile)

        except Exception as exc:
            logger.warning("recruiter_csv row %d: error during extraction: %s", i, exc)

    return profiles


# ---------------------------------------------------------------------------
# Structured: ATS JSON blob
# ---------------------------------------------------------------------------

# Map of ATS field names → canonical field names
_ATS_FIELD_MAP = {
    "applicant_name": "full_name",
    "contact_email": "email",
    "mobile": "phone",
    "employer": "company",
    "position": "title",
    "city": "city",
    "state": "region",
    "country_code": "country",
    "linkedin_url": "linkedin",
    "github_url": "github",
    "tags": "skills",
    "years_exp": "years_experience",
}


def extract_from_ats_json(json_text: str) -> list[CandidateProfile]:
    """Parse an ATS JSON blob (list of applicant objects with non-canonical keys)."""
    source = "ats_json"
    profiles: list[CandidateProfile] = []

    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning("ats_json: invalid JSON: %s", exc)
        return []

    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        logger.warning("ats_json: expected list or object at root, got %s", type(raw).__name__)
        return []

    for i, record in enumerate(raw):
        try:
            if not isinstance(record, dict):
                continue

            # Remap fields
            name = (record.get("applicant_name") or "").strip() or None
            email_raw = (record.get("contact_email") or "").strip() or None
            email, email_ok = validate_email(email_raw)
            if email_raw and not email_ok:
                logger.debug("ats_json record %d: malformed email '%s', dropping", i, email_raw)
            phone_raw = record.get("mobile")
            company = (record.get("employer") or "").strip() or None
            title = (record.get("position") or "").strip() or None
            city = (record.get("city") or "").strip() or None
            region = (record.get("state") or "").strip() or None
            country_raw = record.get("country_code")
            linkedin_raw = record.get("linkedin_url")
            github_raw = record.get("github_url")
            tags = record.get("tags") or []
            years_exp_raw = record.get("years_exp")

            if not name and not email:
                continue

            cid = _id_from_name(name or email or f"ats_{i}")
            prov: list[Provenance] = []

            # Phone
            phone_str = str(phone_raw).strip() if phone_raw else None
            phone_e164, phone_ok = normalize_phone(phone_str)
            if not phone_ok and phone_str:
                logger.debug("ats_json record %d: phone '%s' not normalizable", i, phone_str)

            # Country
            country_code, _ = normalize_country(country_raw)

            # Skills
            skills: list[Skill] = []
            if isinstance(tags, list):
                for tag in tags:
                    if tag:
                        canonical, is_known = canonicalize_skill(str(tag))
                        if canonical:
                            conf = 0.8 if is_known else 0.6
                            skills.append(Skill(name=canonical, confidence=conf, sources=[source]))

            # Years experience
            yoe = _parse_years_experience(years_exp_raw)

            # Links
            links = Links(
                linkedin=clean_url(linkedin_raw),
                github=clean_url(github_raw),
            )

            # Location
            location = None
            if city or region or country_code:
                location = Location(city=city, region=region, country=country_code)

            # Experience
            experience: list[Experience] = []
            if company or title:
                experience.append(Experience(company=company, title=title))

            # Provenance
            if name:
                prov.append(_prov("full_name", source, "remapped"))
            if email:
                prov.append(_prov("emails", source, "remapped"))
            if phone_ok:
                prov.append(_prov("phones", source, "remapped+normalized"))
            if location:
                prov.append(_prov("location", source, "remapped"))
            if skills:
                prov.append(_prov("skills", source, "remapped+canonicalized"))
            if yoe is not None:
                prov.append(_prov("years_experience", source, "remapped"))
            if links.linkedin or links.github:
                prov.append(_prov("links", source, "remapped"))

            profile = CandidateProfile(
                candidate_id=cid,
                full_name=name,
                emails=[email] if email else [],
                phones=[phone_e164] if phone_ok else [],
                location=location,
                links=links,
                years_experience=yoe,
                skills=skills,
                experience=experience,
                provenance=prov,
            )
            profiles.append(profile)

        except Exception as exc:
            logger.warning("ats_json record %d: unexpected error: %s", i, exc)

    return profiles


# ---------------------------------------------------------------------------
# Unstructured: GitHub API JSON
# ---------------------------------------------------------------------------

def extract_from_github_api(json_text: str, candidate_name_hint: str = "") -> CandidateProfile | None:
    """
    Extract candidate data from a GitHub REST API profile response.
    Handles missing fields gracefully.
    """
    source = "github_api"

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning("github_api: invalid JSON: %s", exc)
        return None

    if not isinstance(data, dict):
        return None

    name = data.get("name") or candidate_name_hint or None
    bio = data.get("bio") or None
    location_raw = data.get("location") or None
    email_raw = data.get("email") or None
    email, email_ok = validate_email(email_raw)
    if email_raw and not email_ok:
        logger.debug("github_api: malformed email '%s', dropping", email_raw)
    blog = data.get("blog") or None
    login = data.get("login") or None
    top_langs = data.get("top_languages") or []
    repos = data.get("repos") or []

    prov: list[Provenance] = []

    # Build skills from languages
    skills: list[Skill] = []
    lang_sources: set[str] = set()
    for lang in top_langs:
        canonical, is_known = canonicalize_skill(str(lang))
        if canonical and canonical not in lang_sources:
            skills.append(Skill(name=canonical, confidence=0.7, sources=[source]))
            lang_sources.add(canonical)

    # Also scan repo descriptions for skills
    repo_skill_names = set(s.name for s in skills)
    for repo in repos:
        lang = repo.get("language")
        if lang:
            canonical, _ = canonicalize_skill(lang)
            if canonical and canonical not in repo_skill_names:
                skills.append(Skill(name=canonical, confidence=0.6, sources=[source]))
                repo_skill_names.add(canonical)
        desc = repo.get("description") or ""
        for s in _extract_skills_from_text(desc, source):
            if s.name not in repo_skill_names:
                skills.append(s)
                repo_skill_names.add(s.name)

    # Also extract skills from bio
    if bio:
        for s in _extract_skills_from_text(bio, source):
            if s.name not in repo_skill_names:
                skills.append(s)
                repo_skill_names.add(s.name)

    # Location
    location = None
    if location_raw:
        parts = [p.strip() for p in location_raw.split(",")]
        city = parts[0] if parts else None
        region = parts[1] if len(parts) > 1 else None
        country = None
        if len(parts) > 2:
            country, _ = normalize_country(parts[-1])
        elif len(parts) == 2:
            country, ok = normalize_country(parts[1])
            if not ok:
                region = parts[1]
                country = None
        location = Location(city=city, region=region, country=country)

    # Links
    github_url = f"https://github.com/{login}" if login else None
    portfolio = clean_url(blog) if blog and not blog.startswith("http") else blog or None
    links = Links(github=github_url, portfolio=portfolio)

    # Provenance
    if name:
        prov.append(_prov("full_name", source, "api"))
    if skills:
        prov.append(_prov("skills", source, "api+inferred"))
    if location:
        prov.append(_prov("location", source, "api"))
    if links.github:
        prov.append(_prov("links.github", source, "api"))
    if links.portfolio:
        prov.append(_prov("links.portfolio", source, "api"))
    if email:
        prov.append(_prov("emails", source, "api"))
    if bio:
        prov.append(_prov("headline", source, "api"))

    cid = _id_from_name(name or login or "unknown_github")

    return CandidateProfile(
        candidate_id=cid,
        full_name=name,
        emails=[email] if email else [],
        phones=[],
        location=location,
        links=links,
        headline=bio,
        skills=skills,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Unstructured: Recruiter notes (free text)
# ---------------------------------------------------------------------------

def extract_from_recruiter_notes(text: str) -> CandidateProfile | None:
    """
    Heuristic NLP-lite extraction from free-text recruiter notes.
    Uses regex patterns; never crashes, always marks low confidence.
    """
    source = "recruiter_notes"
    if not text or not text.strip():
        return None

    prov: list[Provenance] = []

    # Name
    name_match = re.search(r"Candidate:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*(?:\n|$)", text)
    name = name_match.group(1) if name_match else None

    # Email
    email_matches = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    emails = list(dict.fromkeys(email_matches))  # deduplicate, preserve order

    # Phone
    phone_matches = re.findall(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    phones: list[str] = []
    for p in phone_matches:
        e164, ok = normalize_phone(p)
        if ok and e164 not in phones:
            phones.append(e164)

    # Skills - both from keyword matching in full text AND explicit Skills: line
    skills_map: dict[str, Skill] = {}
    
    # First check for explicit "Skills: ..." line
    skills_line_match = re.search(r"[Ss]kills[:\s]+([A-Za-z, +#.]+)", text)
    if skills_line_match:
        raw_skills = skills_line_match.group(1).split(",")
        for raw in raw_skills:
            canonical, is_known = canonicalize_skill(raw.strip())
            if canonical:
                conf = 0.8 if is_known else 0.6
                skills_map[canonical] = Skill(name=canonical, confidence=conf, sources=[source])
    
    # Also extract from full text body
    for s in _extract_skills_from_text(text, source):
        if s.name not in skills_map:
            skills_map[s.name] = s

    skills = list(skills_map.values())

    # Education
    education: list[Education] = []
    edu_match = re.search(
        r"(?:BS|MS|PhD|BA|MA|B\.S\.|M\.S\.|B\.A\.|M\.A\.)[.\s]+([A-Za-z ]+),\s*([A-Za-z ]+(?:University|College|Institute|School)),\s*(\d{4})",
        text, re.IGNORECASE
    )
    if edu_match:
        field_of_study = edu_match.group(1).strip()
        institution = edu_match.group(2).strip()
        end_year = int(edu_match.group(3))
        # Detect degree type
        deg_match = re.match(r"(BS|MS|PhD|BA|MA|B\.S\.|M\.S\.|B\.A\.|M\.A\.)", edu_match.group(0), re.IGNORECASE)
        degree = deg_match.group(1).upper().rstrip(".") if deg_match else None
        education.append(Education(
            institution=institution, degree=degree,
            field_of_study=field_of_study, end_year=end_year
        ))

    # Years of experience
    yoe = None
    yoe_match = re.search(r"(\d+(?:\.\d+)?)\s+years?\s+(?:of\s+)?(?:experience|exp\b)", text, re.IGNORECASE)
    if yoe_match:
        try:
            yoe = float(yoe_match.group(1))
        except ValueError:
            pass

    # Location (rough: "Based in City, State")
    location = None
    loc_match = re.search(r"[Bb]ased in ([A-Z][a-zA-Z ]+),\s*([A-Z]{2})\b", text)
    if loc_match:
        location = Location(city=loc_match.group(1).strip(), region=loc_match.group(2))

    # Website / portfolio
    web_matches = re.findall(r"https?://[^\s,]+", text)
    non_email_urls = [u for u in web_matches if "@" not in u]

    links = None
    if non_email_urls:
        links = Links(portfolio=non_email_urls[0] if non_email_urls else None,
                      other=non_email_urls[1:])

    # Provenance
    if name:
        prov.append(_prov("full_name", source, "regex"))
    if emails:
        prov.append(_prov("emails", source, "regex"))
    if phones:
        prov.append(_prov("phones", source, "regex+normalized"))
    if skills:
        prov.append(_prov("skills", source, "keyword_match"))
    if education:
        prov.append(_prov("education", source, "regex"))
    if yoe is not None:
        prov.append(_prov("years_experience", source, "regex"))
    if location:
        prov.append(_prov("location", source, "regex"))

    cid = _id_from_name(name or (emails[0] if emails else "unknown_notes"))

    return CandidateProfile(
        candidate_id=cid,
        full_name=name,
        emails=emails,
        phones=phones,
        location=location,
        links=links,
        years_experience=yoe,
        skills=skills,
        education=education,
        provenance=prov,
    )
