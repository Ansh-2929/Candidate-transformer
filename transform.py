#!/usr/bin/env python3
"""
transform.py - CLI entry point for the Multi-Source Candidate Data Transformer.

Usage:
    python transform.py --help
    python transform.py \\
        --csv  sample_inputs/recruiter_export.csv \\
        --ats  sample_inputs/ats_export.json \\
        --github sample_inputs/github_anshgarg.json --github-name "Ansh Garg" \\
        --notes  sample_inputs/recruiter_notes_ansh.txt \\
        --config configs/default_config.json \\
        --out    output/candidates.json

Exit codes:
    0 - success (even with validation warnings)
    1 - fatal error (bad args, unreadable files, etc.)
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from any directory by adding src/ to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import run_pipeline


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(levelname)s [%(name)s] %(message)s",
        level=level,
        stream=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="transform",
        description="Multi-Source Candidate Data Transformer — Eightfold Engineering Assignment",
    )

    # Input sources
    p.add_argument(
        "--csv",
        metavar="FILE",
        help="Recruiter CSV export file",
    )
    p.add_argument(
        "--ats",
        metavar="FILE",
        help="ATS JSON blob file",
    )
    p.add_argument(
        "--github",
        metavar="FILE",
        action="append",
        default=[],
        help="GitHub API JSON file (repeatable). Pair each with --github-name.",
    )
    p.add_argument(
        "--github-name",
        metavar="NAME",
        action="append",
        default=[],
        dest="github_names",
        help="Candidate name hint for the corresponding --github file.",
    )
    p.add_argument(
        "--notes",
        metavar="FILE",
        action="append",
        default=[],
        help="Recruiter notes text file (repeatable).",
    )

    # Config / output
    p.add_argument(
        "--config",
        metavar="FILE",
        help="JSON output config file for custom projection (optional).",
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        help="Write JSON output to FILE instead of stdout.",
    )
    p.add_argument(
        "--on-missing",
        choices=["null", "omit", "error"],
        default="null",
        help="What to do when a field is missing: null (default), omit, or error.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Emit the full canonical schema (ignores --config).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging to stderr.",
    )

    return p.parse_args()


def build_sources(args: argparse.Namespace) -> list[dict]:
    sources: list[dict] = []

    if args.csv:
        sources.append({"type": "recruiter_csv", "path": args.csv})

    if args.ats:
        sources.append({"type": "ats_json", "path": args.ats})

    # GitHub files + optional name hints
    github_names = args.github_names or []
    for i, gh_file in enumerate(args.github or []):
        hint = github_names[i] if i < len(github_names) else ""
        sources.append({"type": "github_api", "path": gh_file, "name_hint": hint})

    for notes_file in args.notes or []:
        sources.append({"type": "recruiter_notes", "path": notes_file})

    return sources


def load_config(args: argparse.Namespace) -> dict | None:
    if args.full:
        return None
    if not args.config:
        return None
    try:
        return json.loads(Path(args.config).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read config '{args.config}': {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    sources = build_sources(args)
    if not sources:
        print(
            "ERROR: no input sources specified. Use --csv, --ats, --github, or --notes.\n"
            "       Run with --help for usage.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = load_config(args)

    result = run_pipeline(sources, config=config, on_missing=args.on_missing)
    output_json = result.to_json(indent=2)

    # Save output (default: output/candidates.json)
    out_path = Path(args.out) if args.out else Path("output/candidates.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output_json, encoding="utf-8")

    print(output_json)
    print(f"Output written to {out_path}", file=sys.stderr)

    # Print validation warnings to stderr
    if result.validation_errors:
        print("\n⚠  Validation warnings:", file=sys.stderr)
        for cid, errs in result.validation_errors.items():
            for err in errs:
                print(f"  [{cid}] {err}", file=sys.stderr)

    print(
        f"\n✓  {result.n_merged} candidate(s) processed from {result.n_sources} source(s) "
        f"in {result.elapsed_ms:.0f} ms",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
