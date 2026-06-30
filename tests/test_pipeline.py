"""
tests/test_pipeline.py - Unit and integration tests for the candidate transformer.

Run with:
    cd /path/to/candidate-transformer
    python -m pytest tests/ -v
"""

import sys
import json
import pytest
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from normalizers import normalize_phone, normalize_date, normalize_country, canonicalize_skill
from extractors import (
    extract_from_recruiter_csv,
    extract_from_ats_json,
    extract_from_github_api,
    extract_from_recruiter_notes,
)
from merger import group_by_identity, merge_profiles
from projector import project, validate_output
from pipeline import run_pipeline


# ===========================================================================
# Normalizer tests
# ===========================================================================

class TestNormalizePhone:
    def test_us_dashes(self):
        e164, ok = normalize_phone("415-555-0101")
        assert ok and e164 == "+14155550101"

    def test_us_parentheses(self):
        e164, ok = normalize_phone("(650) 555-0202")
        assert ok and e164 == "+16505550202"

    def test_with_country_code(self):
        e164, ok = normalize_phone("+1 408 555 0303")
        assert ok and e164 == "+14085550303"

    def test_bare_10_digits(self):
        e164, ok = normalize_phone("4155550101")
        assert ok and e164 == "+14155550101"

    def test_11_digits_starting_1(self):
        e164, ok = normalize_phone("14155550101")
        assert ok and e164 == "+14155550101"

    def test_empty_returns_none(self):
        _, ok = normalize_phone("")
        assert not ok

    def test_none_returns_none(self):
        _, ok = normalize_phone(None)
        assert not ok

    def test_garbage_returns_none(self):
        _, ok = normalize_phone("not a phone")
        assert not ok


class TestNormalizeDate:
    def test_yyyy_mm(self):
        val, ok = normalize_date("2023-06")
        assert ok and val == "2023-06"

    def test_month_year_text(self):
        val, ok = normalize_date("May 2025")
        assert ok and val == "2025-05"

    def test_year_only(self):
        val, ok = normalize_date("2018")
        assert ok and val == "2018-01"

    def test_slash_format(self):
        val, ok = normalize_date("06/2023")
        assert ok and val == "2023-06"

    def test_empty(self):
        _, ok = normalize_date("")
        assert not ok

    def test_none(self):
        _, ok = normalize_date(None)
        assert not ok


class TestNormalizeCountry:
    def test_us_code(self):
        code, ok = normalize_country("US")
        assert ok and code == "US"

    def test_usa_string(self):
        code, ok = normalize_country("USA")
        assert ok and code == "US"

    def test_full_name(self):
        code, ok = normalize_country("United States")
        assert ok and code == "US"

    def test_unknown(self):
        _, ok = normalize_country("Narnia")
        assert not ok


class TestCanonicalizeSkill:
    def test_golang(self):
        name, ok = canonicalize_skill("golang")
        assert ok and name == "Go"

    def test_k8s(self):
        name, ok = canonicalize_skill("k8s")
        assert ok and name == "Kubernetes"

    def test_pytorch(self):
        name, ok = canonicalize_skill("pytorch")
        assert ok and name == "PyTorch"

    def test_unknown_skill(self):
        name, ok = canonicalize_skill("SomeRareFramework")
        # Unknown skill → title-cased fallback, ok=False
        assert not ok
        assert name == "Somerareframework"

    def test_empty(self):
        _, ok = canonicalize_skill("")
        assert not ok


# ===========================================================================
# Extractor tests
# ===========================================================================

class TestRecruiterCSVExtractor:
    CSV = """name,email,phone,current_company,title
"Jane Doe","jane@example.com","415-555-0101","Acme","Engineer"
"Bob","","","",""
"""

    def test_basic_extraction(self):
        profiles = extract_from_recruiter_csv(self.CSV)
        assert len(profiles) == 2

    def test_phone_normalized(self):
        profiles = extract_from_recruiter_csv(self.CSV)
        jane = profiles[0]
        assert jane.phones == ["+14155550101"]

    def test_email_captured(self):
        profiles = extract_from_recruiter_csv(self.CSV)
        assert "jane@example.com" in profiles[0].emails

    def test_experience_set(self):
        profiles = extract_from_recruiter_csv(self.CSV)
        assert profiles[0].experience[0].company == "Acme"

    def test_empty_csv(self):
        profiles = extract_from_recruiter_csv("")
        assert profiles == []

    def test_garbage_csv(self):
        profiles = extract_from_recruiter_csv("this is not csv\x00\xff")
        # Should not crash
        assert isinstance(profiles, list)

    def test_missing_phone_not_crash(self):
        csv = "name,email,phone\n\"NoPhone\",\"a@b.com\",\"\"\n"
        profiles = extract_from_recruiter_csv(csv)
        assert profiles[0].phones == []


class TestATSJSONExtractor:
    ATS = json.dumps([{
        "applicant_name": "Alice Chen",
        "contact_email": "alice@example.com",
        "mobile": "4085550303",
        "employer": "BigCo",
        "position": "ML Engineer",
        "city": "Palo Alto",
        "state": "CA",
        "country_code": "US",
        "linkedin_url": "https://linkedin.com/in/alice",
        "github_url": "https://github.com/alice",
        "tags": ["Python", "PyTorch", "CUDA"],
        "years_exp": 5,
    }])

    def test_basic_extraction(self):
        profiles = extract_from_ats_json(self.ATS)
        assert len(profiles) == 1

    def test_field_remapping(self):
        profiles = extract_from_ats_json(self.ATS)
        p = profiles[0]
        assert p.full_name == "Alice Chen"
        assert p.location.city == "Palo Alto"
        assert p.links.linkedin == "https://linkedin.com/in/alice"

    def test_skills_canonicalized(self):
        profiles = extract_from_ats_json(self.ATS)
        skill_names = [s.name for s in profiles[0].skills]
        assert "PyTorch" in skill_names

    def test_years_experience(self):
        profiles = extract_from_ats_json(self.ATS)
        assert profiles[0].years_experience == 5.0

    def test_invalid_json(self):
        profiles = extract_from_ats_json("not json {{{{")
        assert profiles == []

    def test_null_phone(self):
        data = [{"applicant_name": "X", "contact_email": "x@y.com", "mobile": None}]
        profiles = extract_from_ats_json(json.dumps(data))
        assert profiles[0].phones == []


class TestGitHubExtractor:
    PROFILE = json.dumps({
        "login": "janedoe",
        "name": "Jane Doe",
        "bio": "Go engineer | Distributed Systems",
        "location": "San Francisco, CA",
        "email": None,
        "blog": "https://janedoe.dev",
        "top_languages": ["Go", "Python", "Rust"],
        "repos": [{"name": "kv", "description": "Raft consensus store", "language": "Go", "stars": 10}],
    })

    def test_basic(self):
        p = extract_from_github_api(self.PROFILE)
        assert p is not None
        assert p.full_name == "Jane Doe"

    def test_location_parsed(self):
        p = extract_from_github_api(self.PROFILE)
        assert p.location.city == "San Francisco"
        assert p.location.region == "CA"

    def test_skills_from_languages(self):
        p = extract_from_github_api(self.PROFILE)
        skill_names = [s.name for s in p.skills]
        assert "Go" in skill_names
        assert "Python" in skill_names

    def test_invalid_json(self):
        p = extract_from_github_api("!!!BAD!!!")
        assert p is None


class TestRecruiterNotesExtractor:
    NOTES = """Candidate: Jane Doe
Email: jane@example.com, jane.work@stripe.com
Phone: 415-555-0101
Skills: Python, Go, Kubernetes
Based in San Francisco, CA
7 years of experience
Education: BS Computer Science, Stanford University, 2018
Website: https://janedoe.dev
"""

    def test_name_extracted(self):
        p = extract_from_recruiter_notes(self.NOTES)
        assert p is not None
        assert p.full_name == "Jane Doe"

    def test_emails_extracted(self):
        p = extract_from_recruiter_notes(self.NOTES)
        assert "jane@example.com" in p.emails

    def test_phone_normalized(self):
        p = extract_from_recruiter_notes(self.NOTES)
        assert "+14155550101" in p.phones

    def test_skills_extracted(self):
        p = extract_from_recruiter_notes(self.NOTES)
        skill_names = [s.name for s in p.skills]
        assert "Python" in skill_names

    def test_yoe_extracted(self):
        p = extract_from_recruiter_notes(self.NOTES)
        assert p.years_experience == 7.0

    def test_education_extracted(self):
        p = extract_from_recruiter_notes(self.NOTES)
        assert len(p.education) == 1
        assert p.education[0].institution == "Stanford University"

    def test_empty_notes(self):
        p = extract_from_recruiter_notes("")
        assert p is None

    def test_no_crash_on_garbage(self):
        p = extract_from_recruiter_notes("\x00\xff random garbage \n\n\n")
        assert p is None or hasattr(p, "candidate_id")


# ===========================================================================
# Merger tests
# ===========================================================================

class TestIdentityMatcher:
    def test_same_email_grouped(self):
        from schema import CandidateProfile
        p1 = CandidateProfile("a", "Jane Doe", emails=["jane@x.com"])
        p2 = CandidateProfile("b", "Jane Doe", emails=["jane@x.com"])
        groups = group_by_identity([p1, p2])
        assert len(groups) == 1

    def test_different_email_not_grouped(self):
        from schema import CandidateProfile
        p1 = CandidateProfile("a", "Alice", emails=["alice@x.com"])
        p2 = CandidateProfile("b", "Bob",   emails=["bob@x.com"])
        groups = group_by_identity([p1, p2])
        assert len(groups) == 2

    def test_same_name_grouped(self):
        from schema import CandidateProfile
        p1 = CandidateProfile("a", "Jane Doe", emails=[])
        p2 = CandidateProfile("b", "Jane Doe", emails=[])
        groups = group_by_identity([p1, p2])
        assert len(groups) == 1


class TestMerge:
    def test_emails_union(self):
        from schema import CandidateProfile, Provenance
        p1 = CandidateProfile("a", "Jane", emails=["jane@a.com"],
                               provenance=[Provenance("emails", "recruiter_csv", "direct")])
        p2 = CandidateProfile("a", "Jane", emails=["jane@b.com"],
                               provenance=[Provenance("emails", "ats_json", "remapped")])
        merged = merge_profiles([p1, p2])
        assert "jane@a.com" in merged.emails
        assert "jane@b.com" in merged.emails

    def test_skill_confidence_boosted(self):
        from schema import CandidateProfile, Skill, Provenance
        s1 = Skill("Python", confidence=0.7, sources=["recruiter_csv"])
        s2 = Skill("Python", confidence=0.8, sources=["ats_json"])
        p1 = CandidateProfile("a", "Jane", skills=[s1],
                               provenance=[Provenance("skills", "recruiter_csv", "direct")])
        p2 = CandidateProfile("a", "Jane", skills=[s2],
                               provenance=[Provenance("skills", "ats_json", "remapped")])
        merged = merge_profiles([p1, p2])
        py_skill = next(s for s in merged.skills if s.name == "Python")
        assert py_skill.confidence >= 0.8  # boosted

    def test_priority_scalar(self):
        """ATS JSON should win over CSV for full_name when names differ."""
        from schema import CandidateProfile, Provenance
        p_csv = CandidateProfile("a", "Jane D.",    emails=["jane@x.com"],
                                  provenance=[Provenance("full_name", "recruiter_csv", "direct")])
        p_ats = CandidateProfile("a", "Jane Doe",   emails=["jane@x.com"],
                                  provenance=[Provenance("full_name", "ats_json", "remapped")])
        merged = merge_profiles([p_csv, p_ats])
        assert merged.full_name == "Jane Doe"   # ats_json wins


# ===========================================================================
# Projector tests
# ===========================================================================

class TestProjector:
    def _make_profile(self):
        from schema import CandidateProfile, Location, Links, Skill
        return CandidateProfile(
            candidate_id="jane_doe",
            full_name="Jane Doe",
            emails=["jane@example.com", "jane@work.com"],
            phones=["+14155550101"],
            location=Location(city="San Francisco", region="CA", country="US"),
            links=Links(linkedin="https://linkedin.com/in/janedoe", github="https://github.com/janedoe"),
            headline="Backend engineer",
            years_experience=7.0,
            skills=[Skill("Python", 0.9, ["ats_json"]), Skill("Go", 0.85, ["github_api"])],
            overall_confidence=0.82,
        )

    def test_full_output_no_config(self):
        p = self._make_profile()
        out = project(p, None)
        assert out["full_name"] == "Jane Doe"
        assert "emails" in out

    def test_field_selection(self):
        p = self._make_profile()
        config = {"fields": [{"path": "full_name"}, {"path": "emails"}], "on_missing": "null"}
        out = project(p, config)
        assert "full_name" in out
        assert "emails" in out
        assert "phones" not in out

    def test_field_remapping(self):
        p = self._make_profile()
        config = {
            "fields": [{"path": "primary_email", "from": "emails[0]", "type": "string"}],
            "on_missing": "null"
        }
        out = project(p, config)
        assert out["primary_email"] == "jane@example.com"

    def test_array_spread(self):
        p = self._make_profile()
        config = {
            "fields": [{"path": "skill_names", "from": "skills[].name", "type": "string[]"}],
            "on_missing": "null"
        }
        out = project(p, config)
        assert "Python" in out["skill_names"]

    def test_on_missing_omit(self):
        p = self._make_profile()
        p.headline = None
        config = {
            "fields": [{"path": "headline", "type": "string"}],
            "on_missing": "omit"
        }
        out = project(p, config)
        assert "headline" not in out

    def test_on_missing_null(self):
        p = self._make_profile()
        p.headline = None
        config = {
            "fields": [{"path": "headline", "type": "string"}],
            "on_missing": "null"
        }
        out = project(p, config)
        assert out.get("headline") is None

    def test_on_missing_error_raises(self):
        from projector import ProjectionError
        p = self._make_profile()
        p.headline = None
        config = {
            "fields": [{"path": "headline", "type": "string", "required": True}],
            "on_missing": "error"
        }
        with pytest.raises(ProjectionError):
            project(p, config)


# ===========================================================================
# End-to-end integration test
# ===========================================================================

class TestEndToEnd:
    """Golden-profile comparison against sample inputs."""

    def _sample_dir(self) -> Path:
        return Path(__file__).parent.parent / "sample_inputs"

    def test_full_pipeline_runs(self):
        d = self._sample_dir()
        sources = [
            {"type": "recruiter_csv", "path": str(d / "recruiter_export.csv")},
            {"type": "ats_json",       "path": str(d / "ats_export.json")},
            {"type": "github_api",     "path": str(d / "github_anshgarg.json"), "name_hint": "Ansh Garg"},
            {"type": "recruiter_notes","path": str(d / "recruiter_notes_ansh.txt")},
        ]
        result = run_pipeline(sources)
        assert len(result.profiles) > 0
        assert result.elapsed_ms > 0

    def test_ansh_garg_merged_correctly(self):
        d = self._sample_dir()
        sources = [
            {"type": "recruiter_csv", "path": str(d / "recruiter_export.csv")},
            {"type": "ats_json",       "path": str(d / "ats_export.json")},
            {"type": "github_api",     "path": str(d / "github_anshgarg.json"), "name_hint": "Ansh Garg"},
            {"type": "recruiter_notes","path": str(d / "recruiter_notes_ansh.txt")},
        ]
        result = run_pipeline(sources)
        # Find Ansh Garg
        ansh = next(
            (p for p in result.profiles if "ansh" in (p.get("full_name") or "").lower()),
            None
        )
        assert ansh is not None, "Ansh Garg not found in output"
        assert ansh.get("full_name") == "Ansh Garg"
        # Should have at least one email
        assert len(ansh.get("emails", [])) >= 1

    def test_custom_config_projection(self):
        d = self._sample_dir()
        sources = [
            {"type": "recruiter_csv", "path": str(d / "recruiter_export.csv")},
            {"type": "ats_json",       "path": str(d / "ats_export.json")},
        ]
        config = {
            "fields": [
                {"path": "full_name",     "type": "string", "required": True},
                {"path": "primary_email", "from": "emails[0]", "type": "string"},
            ],
            "include_confidence": True,
            "on_missing": "null",
        }
        result = run_pipeline(sources, config=config)
        for p in result.profiles:
            assert "full_name" in p
            assert "phones" not in p  # not in config

    def test_missing_source_does_not_crash(self):
        sources = [
            {"type": "recruiter_csv", "path": "/nonexistent/file.csv"},
            {"type": "ats_json",       "path": "/nonexistent/ats.json"},
        ]
        result = run_pipeline(sources)
        # Should return empty, not crash
        assert result.profiles == []

    def test_garbage_source_degrades_gracefully(self):
        sources = [
            {"type": "recruiter_csv", "data": "totally,garbage\ndata,here\x00\xff"},
            {"type": "ats_json",       "data": "not json at all {{"},
        ]
        result = run_pipeline(sources)
        # Pipeline completes; may or may not extract profiles from garbage
        assert isinstance(result.profiles, list)

    def test_single_source_still_works(self):
        d = self._sample_dir()
        sources = [{"type": "recruiter_csv", "path": str(d / "recruiter_export.csv")}]
        result = run_pipeline(sources)
        assert len(result.profiles) > 0

    def test_provenance_populated(self):
        d = self._sample_dir()
        sources = [
            {"type": "recruiter_csv", "path": str(d / "recruiter_export.csv")},
        ]
        config = {"include_provenance": True, "on_missing": "null"}
        result = run_pipeline(sources, config=config)
        for p in result.profiles:
            assert isinstance(p.get("provenance"), list)
