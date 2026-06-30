"""
normalizers.py - Pure normalization helpers (no I/O, no side effects).

All functions return (normalized_value, ok: bool).
On failure they return (None, False) — they never invent data.
"""

from __future__ import annotations
import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Phone → E.164
# ---------------------------------------------------------------------------

_STRIP_PHONE = re.compile(r"[^\d+]")

def normalize_phone(raw: str | None) -> Tuple[Optional[str], bool]:
    """
    Convert messy phone strings to E.164 (+1XXXXXXXXXX).
    Returns (e164, True) on success, (None, False) otherwise.

    Handles:
        "415-555-0101"     → "+14155550101"
        "(650) 555-0202"   → "+16505550202"
        "+1 408 555 0303"  → "+14085550303"
        "4155550101"       → "+14155550101"  (assumes US if no country code)
        "00 44 20 7946 0958" → "+442079460958"
    """
    if not raw or not raw.strip():
        return None, False

    s = raw.strip()
    has_plus = s.startswith("+")

    # Strip everything except digits (the leading '+', if any, is tracked
    # separately via has_plus and must not be kept in `digits`, or it would
    # be duplicated when we rebuild the E.164 string below).
    digits = _STRIP_PHONE.sub("", s).lstrip("+")

    # Remove leading zeros from international dialing (00XXX) - only when no + prefix
    if not has_plus and digits.startswith("00"):
        digits = digits[2:]

    # If exactly 10 digits, assume US/Canada (NANP)
    if len(digits) == 10:
        return f"+1{digits}", True

    # If 11 digits starting with 1, it's US/Canada with country code
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}", True

    # International: must be 7–15 digits (E.164 range)
    if 7 <= len(digits) <= 15:
        return f"+{digits}", True

    return None, False


# ---------------------------------------------------------------------------
# Date → YYYY-MM
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    (re.compile(r"(\d{4})-(\d{2})"), lambda m: f"{m.group(1)}-{m.group(2)}"),
    (re.compile(r"(\d{1,2})/(\d{4})"), lambda m: f"{m.group(2)}-{int(m.group(1)):02d}"),
    (re.compile(r"(\d{4})/(\d{1,2})"), lambda m: f"{m.group(1)}-{int(m.group(2)):02d}"),
    (re.compile(r"(\d{4})$"), lambda m: f"{m.group(1)}-01"),
]

_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

def normalize_date(raw: str | None) -> Tuple[Optional[str], bool]:
    """
    Parse flexible date strings to YYYY-MM.
    Returns (yyyy_mm, True) on success, (None, False) otherwise.
    """
    if not raw:
        return None, False

    s = raw.strip().lower()

    # Handle text months like "May 2025" or "2025 May"
    for abbr, num in _MONTH_MAP.items():
        pattern = re.compile(rf"\b{abbr}\w*\s+(\d{{4}})\b|\b(\d{{4}})\s+{abbr}\w*\b")
        m = pattern.search(s)
        if m:
            year = m.group(1) or m.group(2)
            return f"{year}-{num}", True

    for pattern, formatter in _DATE_PATTERNS:
        m = pattern.search(s)
        if m:
            return formatter(m), True

    return None, False


# ---------------------------------------------------------------------------
# Country → ISO-3166 alpha-2
# ---------------------------------------------------------------------------

_COUNTRY_MAP = {
    "us": "US", "usa": "US", "united states": "US", "united states of america": "US",
    "uk": "GB", "gb": "GB", "united kingdom": "GB", "great britain": "GB",
    "ca": "CA", "canada": "CA",
    "in": "IN", "india": "IN",
    "de": "DE", "germany": "DE",
    "fr": "FR", "france": "FR",
    "au": "AU", "australia": "AU",
    "sg": "SG", "singapore": "SG",
    "jp": "JP", "japan": "JP",
    "cn": "CN", "china": "CN",
    "br": "BR", "brazil": "BR",
    "nl": "NL", "netherlands": "NL",
    "se": "SE", "sweden": "SE",
    "no": "NO", "norway": "NO",
    "ch": "CH", "switzerland": "CH",
    "nz": "NZ", "new zealand": "NZ",
    "ie": "IE", "ireland": "IE",
    "il": "IL", "israel": "IL",
    "kr": "KR", "south korea": "KR",
}

def normalize_country(raw: str | None) -> Tuple[Optional[str], bool]:
    """Normalise to ISO-3166 alpha-2. Returns (code, True) or (None, False)."""
    if not raw:
        return None, False
    key = raw.strip().lower()
    code = _COUNTRY_MAP.get(key)
    if code:
        return code, True
    # If already a 2-letter uppercase code, trust it
    if re.fullmatch(r"[A-Z]{2}", raw.strip()):
        return raw.strip(), True
    return None, False


# ---------------------------------------------------------------------------
# Skill names → canonical
# ---------------------------------------------------------------------------

_SKILL_ALIASES: dict[str, str] = {
    # Python
    "python3": "Python", "python 3": "Python", "py": "Python",
    # JavaScript
    "javascript": "JavaScript", "js": "JavaScript", "ecmascript": "JavaScript",
    "node.js": "Node.js", "nodejs": "Node.js",
    "typescript": "TypeScript", "ts": "TypeScript",
    # Go
    "golang": "Go",
    # Java
    "java8": "Java", "java 8": "Java",
    # C++
    "c++": "C++", "cpp": "C++",
    # Machine Learning
    "ml": "Machine Learning", "machine learning": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "pytorch": "PyTorch", "torch": "PyTorch",
    "tensorflow": "TensorFlow", "tf": "TensorFlow",
    "llms": "LLMs", "large language models": "LLMs",
    "cuda": "CUDA",
    # Infrastructure
    "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "docker": "Docker",
    "kafka": "Kafka", "apache kafka": "Kafka",
    "aws": "AWS", "amazon web services": "AWS",
    "gcp": "GCP", "google cloud": "GCP", "google cloud platform": "GCP",
    "azure": "Azure", "microsoft azure": "Azure",
    # Databases
    "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "sql": "SQL",
    # Protocols/APIs
    "grpc": "gRPC",
    "rest": "REST",
    "graphql": "GraphQL",
    # Frontend / full-stack
    "react.js": "React.js", "reactjs": "React.js", "react js": "React.js", "react": "React.js",
    "express.js": "Express.js", "expressjs": "Express.js", "express js": "Express.js", "express": "Express.js",
    "vue.js": "Vue.js", "vuejs": "Vue.js",
    "next.js": "Next.js", "nextjs": "Next.js",
    "tailwind": "Tailwind CSS", "tailwindcss": "Tailwind CSS",
    # Security tools
    "wireshark": "Wireshark",
    "openvas": "OpenVAS",
    "nmap": "Nmap",
    "metasploit": "Metasploit",
    "burpsuite": "Burp Suite", "burp suite": "Burp Suite", "burp": "Burp Suite",
    # DevOps / tooling
    "git": "Git",
    "linux": "Linux",
    "ci/cd": "CI/CD", "cicd": "CI/CD",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    # Other
    "distributed systems": "Distributed Systems",
    "rust": "Rust",
    "shell": "Shell/Bash", "bash": "Shell/Bash",
    "java": "Java",
    "python": "Python",
    "go": "Go",
}


def canonicalize_skill(raw: str | None) -> Tuple[Optional[str], bool]:
    """
    Map a raw skill name to its canonical form.
    Returns (canonical_name, True) or (raw.title(), False) as a soft fallback.
    """
    if not raw or not raw.strip():
        return None, False
    key = raw.strip().lower()
    if key in _SKILL_ALIASES:
        return _SKILL_ALIASES[key], True
    # Fallback: title-case the raw value, flag as not canonicalized
    return raw.strip().title(), False


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def clean_url(raw: str | None) -> Optional[str]:
    """Strip whitespace, ensure https:// prefix. Returns None for empty."""
    if not raw or not raw.strip():
        return None
    url = raw.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def extract_github_username(url: str | None) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"github\.com/([A-Za-z0-9_.-]+)", url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Email validation
# ---------------------------------------------------------------------------

# Deliberately simple: catches the common malformed cases (missing @,
# missing domain, stray whitespace) without trying to fully implement
# RFC 5322. Good enough to stop obvious garbage from being trusted as a
# contact email; not a substitute for actually verifying deliverability.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def validate_email(raw: str | None) -> Tuple[Optional[str], bool]:
    """
    Lightly validate and normalize an email address (trim + lowercase).
    Returns (email, True) if it looks structurally valid, (None, False)
    otherwise. Never raises.
    """
    if not raw or not raw.strip():
        return None, False
    candidate = raw.strip().lower()
    if _EMAIL_RE.match(candidate):
        return candidate, True
    return None, False
