# Multi-Source Candidate Data Transformer

A pipeline that ingests messy candidate data from multiple sources (recruiter
CSV exports, ATS JSON blobs, GitHub API responses, free-text recruiter notes),
normalizes and reconciles it into one canonical profile per person, and
projects that profile into whatever shape a downstream consumer needs.

## Why it's built this way

Real candidate data never agrees with itself. A recruiter's CSV might have a
clean phone number while the ATS has a stale one; GitHub knows nothing about
phone numbers at all but is the best source for skills inferred from actual
repos. The pipeline is built around one idea: **never throw information away,
and never guess silently**. Every value that makes it into the final profile
can be traced back to where it came from and how confident we are in it.

## Architecture

```
sources (csv / json / txt)
        │
        ▼
┌───────────────┐
│  extractors.py │  source-specific parsing → partial CandidateProfile objects
└───────┬────────┘  (normalizes phones/dates/countries/skills as it goes)
        │
        ▼
┌───────────────┐
│   merger.py    │  groups partials by identity (shared email / matching name),
└───────┬────────┘  merges fields by source-trust priority, computes confidence
        │
        ▼
┌───────────────┐
│  projector.py  │  shapes the canonical profile into a config-defined output
└───────┬────────┘  (dot-path + array-index field selection, type coercion)
        │
        ▼
┌───────────────┐
│  pipeline.py   │  orchestrates the above, collects validation warnings
└───────┬────────┘
        │
        ▼
┌───────────────┐
│ transform.py   │  CLI entry point
└────────────────┘
```

### `schema.py` — the canonical model
A `CandidateProfile` dataclass is the single source of truth for what a
"complete" candidate record looks like. Every extractor builds toward this
shape; nothing downstream needs to know which source a record came from.

### `normalizers.py` — pure functions, no I/O
Phone → E.164, free-text dates → `YYYY-MM`, country names → ISO-3166 alpha-2,
skill aliases → a canonical skill vocabulary. Each function returns
`(value, ok)` instead of raising or guessing — a normalizer that can't
confidently parse its input returns `(None, False)` rather than fabricating a
plausible-looking but wrong value. This is the load-bearing design decision
in the whole project: **garbage in should produce an honest "I don't know,"
not a confident-looking lie.**

### `extractors.py` — one function per source type
Each extractor is defensive by construction: malformed rows/records are
logged and skipped, never allowed to crash the whole batch. Emails are run
through a lightweight structural validator (`validate_email`) before being
trusted — a malformed address from a messy CSV is dropped with a debug log
line rather than silently propagated downstream. Recruiter notes use
targeted regexes (e.g. `Candidate: Ansh Garg`, a `Skills:` line) plus a
keyword scan against the known-skills dictionary, each carrying a different
confidence score depending on how directly it was stated.

### `merger.py` — identity resolution + conflict resolution
- **Identity**: union-find groups partial profiles that share an email or
  have a matching normalized name. This handles the case where the same
  person appears once in the CSV and once in the ATS blob.
- **Conflicts**: scalar fields (name, headline, location, years experience)
  are resolved by source trust priority — `ats_json > recruiter_csv >
  github_api > recruiter_notes` — because structured systems of record are
  more trustworthy than free-text notes. List fields (emails, phones,
  skills, experience, education) are unioned and deduplicated instead of
  picking a winner, since there's no reason to discard a second valid email.
- **Confidence**: a single 0–1 score per candidate, weighted by source
  diversity (more independent sources agreeing = more trust), core field
  coverage, and average skill confidence.
- **Logging**: each merge logs how many identity clusters were formed and,
  per candidate, how many sources contributed and the resulting confidence
  score — enough to debug a bad merge without needing a debugger.

### `projector.py` — config-driven output shaping
Different consumers want different shapes of the same data — a recruiter UI
wants a flat card, an ML pipeline wants just skill vectors. Rather than
hardcoding output formats, a JSON config describes a list of `{path, from,
type, required, normalize}` field specs, and the projector walks the
canonical dict accordingly. Supports plain paths (`location.city`), array
indexing (`emails[0]`), array spread (`skills[].name`), and indexed-then-
nested paths (`experience[0].title`).

### `pipeline.py` — orchestration
Ties extraction → identity grouping → merging → projection → validation
together, and degrades gracefully at every stage (a bad source is logged and
skipped, not fatal; a failed required-field projection is caught per-
candidate so one bad record doesn't kill the whole batch).

## Bugs found and fixed during review

1. **`normalize_phone` double-plus bug.** The digit-stripping regex
   (`[^\d+]`) preserved the `+` character, which was then re-added when
   rebuilding the E.164 string — so any already-`+`-prefixed number (e.g.
   `"+1 408 555 0303"`) came out as `"++14085550303"`. Fixed by stripping the
   leading `+` out of the digit string before rebuilding it.

2. **`projector._get_path` couldn't resolve indexed-then-nested paths.** The
   path-resolution regexes handled a bare index (`emails[0]`) or an array
   spread (`skills[].name`) but never both combined, so config fields like
   `experience[0].title` and `experience[0].company` silently resolved to
   `null` even when the data was present in the merged profile. Fixed by
   extending the index-match regex to optionally capture a trailing dotted
   path and recursing into the indexed element.

## Known limitations

- `normalize_date` doesn't validate that a parsed month is in 1–12 — an
  ambiguous string like `"13/2025"` (day/year vs month/year) will silently
  produce `"2025-13"`. Low blast radius since this only affects malformed
  date strings, but worth fixing with a bounds check if extended further.
- Identity resolution uses exact (normalized) name matching, not fuzzy
  matching — "Jon Smith" and "John Smith" from different sources won't be
  merged. A production version would want a fuzzy-match threshold (e.g.
  Levenshtein distance) with a confidence penalty for fuzzy merges.
- Source trust priority is a fixed global ordering
  (`ats_json > recruiter_csv > github_api > recruiter_notes`). A more
  sophisticated version might make this per-field (e.g. trust GitHub more
  than the ATS for skills, but trust the ATS more for contact info).
- `canonicalize_skill`'s fallback for unrecognized skill names is a naive
  `str.title()` — fine for single words (`"wireshark"` → `"Wireshark"`) but
  produces odd casing for anything with its own internal capitalization
  convention (`"react.js"` → `"React.Js"`, `"sql"` → `"Sql"`). The fix is a
  small list of casing overrides for known-but-uncanonicalized terms, or a
  regex that preserves dotted/acronym segments instead of blindly
  title-casing every word.

## Running it

```bash
# Run tests
python -m pytest tests/ -v

# Run the full pipeline against sample data, recruiter-card output shape
python transform.py \
  --csv sample_inputs/recruiter_export.csv \
  --ats sample_inputs/ats_export.json \
  --github sample_inputs/github_anshgarg.json --github-name "Ansh Garg" \
  --notes sample_inputs/recruiter_notes_ansh.txt \
  --config configs/recruiter_card.json \
  --out output/candidates.json

# Emit the full canonical schema instead of a projected shape
python transform.py --csv sample_inputs/recruiter_export.csv --full
```

The sample inputs describe two candidates spread realistically across the
four source types (no single source has the complete picture, which is the
whole point of the merge step) — one candidate's data is intentionally
incomplete in places (no current company, 0 years experience) to exercise
the pipeline's handling of partial/missing fields, and a second candidate
provides contrast for the identity-resolution logic.

## Project layout

```
candidate-transformer/
├── transform.py          # CLI entry point
├── src/
│   ├── schema.py          # canonical CandidateProfile dataclasses
│   ├── normalizers.py     # pure normalization functions
│   ├── extractors.py      # per-source parsing logic
│   ├── merger.py          # identity resolution + merge + confidence
│   ├── projector.py       # config-driven output shaping
│   └── pipeline.py        # orchestration
├── tests/
│   └── test_pipeline.py   # unit + end-to-end tests
├── sample_inputs/         # example data for manual runs and e2e tests
├── configs/               # example output projection configs
└── output/                # default destination for --out
```
