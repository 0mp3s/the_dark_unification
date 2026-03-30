"""
runner/pipeline.py
==================
Reads docs/execution_pipeline.csv and provides the step list
with input-file existence checks.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT    = Path(__file__).resolve().parent.parent
CSV_PATH     = REPO_ROOT / "docs" / "execution_pipeline.csv"
_ARCHIVE_DIR = REPO_ROOT / "data" / "archive"


@dataclass
class PipelineStep:
    order:        int
    stage:        str
    dependencies: str
    script:       str
    reads:        str          # raw string from CSV
    writes:       str          # raw string from CSV
    paper_section: str

    # resolved at runtime
    missing_inputs: List[str] = field(default_factory=list)
    is_long:        bool = False          # Stage 0 flag
    last_run_at:    Optional[str] = None  # ISO timestamp of most recent output file
    input_stale:    bool = False          # True if any input is newer than latest output

    @property
    def label(self) -> str:
        return Path(self.script).name

    @property
    def stage_number(self) -> str:
        """Return e.g. '0', '1', '7' from 'Stage 0 - VPM Scan'."""
        part = self.stage.split("-")[0].strip()
        return part.split()[-1] if " " in part else part

    def to_dict(self) -> dict:
        return {
            "order":         self.order,
            "stage":         self.stage,
            "stage_number":  self.stage_number,
            "script":        self.script,
            "label":         self.label,
            "reads":         self.reads,
            "writes":        self.writes,
            "dependencies":  self.dependencies,
            "paper_section": self.paper_section,
            "missing_inputs": self.missing_inputs,
            "is_long":       self.is_long,
            "last_run_at":   self.last_run_at,
            "input_stale":   self.input_stale,
        }


def _resolve_archive(token: str) -> List[Path]:
    """
    Resolve data/X.csv → data/archive/X_*.csv (timestamped files).
    Returns matching paths sorted by name (newest last).
    """
    if not token.startswith("data/"):
        return []
    p = Path(token)
    stem = p.stem
    ext  = p.suffix or ".csv"
    return sorted(_ARCHIVE_DIR.glob(f"{stem}_*{ext}"))


def _check_inputs(reads_raw: str, script: str = "") -> List[str]:
    """
    Returns list of paths that are declared in Reads but do not exist.
    Skips entries that are clearly code references (contain no '/' or '.')
    or are 'none'.
    """
    if not reads_raw or reads_raw.lower() in ("none", "nothing (grid scan from scratch)",
                                              "nothing (own grid scan + boltzmann bisection)",
                                              "config params", "embedded data",
                                              "none (imports from #42)"):
        return []

    script_dir = REPO_ROOT / Path(script).parent if script else REPO_ROOT

    missing = []
    # split on comma, strip parens/notes after the actual path
    for token in reads_raw.split(","):
        token = token.strip()
        # remove parenthetical notes
        if "(" in token:
            token = token[:token.index("(")].strip()
        # skip if clearly not a file path
        if not token or " " in token or ("." not in token and "/" not in token):
            continue
        # skip pure code references (no slash, ends in .py but contains no /)
        if token.endswith(".py") and "/" not in token:
            continue
        full = REPO_ROOT / token
        if full.exists():
            continue
        # check archive for timestamped version (data/X.csv → data/archive/X_*.csv)
        if _resolve_archive(token):
            continue
        # check relative to script directory (e.g. gravothermal/dsphs_data.csv)
        if (script_dir / token).exists():
            continue
        # check basename in script directory (CSV may redundantly include subdir)
        if (script_dir / Path(token).name).exists():
            continue
        # check data/ prefix for bare config references (global_config.json)
        if "/" not in token and (REPO_ROOT / "data" / token).exists():
            continue
        missing.append(token)
    return missing


# ── output file parsers ─────────────────────────────────────────────────────

def _parse_file_tokens(raw: str) -> List[str]:
    """Extract plausible file paths from a Writes or Reads cell."""
    skip = {"none", "console", "console only", "publication figure",
            "pdf", "csv", "markdown"}
    tokens = []
    for token in raw.split(","):
        token = token.strip()
        if "(" in token:
            token = token[:token.index("(")].strip()
        token = token.strip()
        if not token or token.lower() in skip:
            continue
        # keep if it has an extension or a /
        if "." in token or "/" in token or token.startswith("output"):
            tokens.append(token)
    return tokens


def _resolve_glob(pattern: str, base: Path) -> List[Path]:
    """Expand a pattern like 'output/*.png' relative to base."""
    if "*" in pattern:
        return list(base.glob(pattern))
    full = base / pattern
    return [full] if full.exists() else []


def _get_output_mtime(step: "PipelineStep") -> Optional[float]:
    """Return the mtime of the most-recently-modified output file, or None."""
    script_dir = REPO_ROOT / Path(step.script).parent
    tokens = _parse_file_tokens(step.writes)
    mtimes: List[float] = []
    for token in tokens:
        # 1. relative to script dir (e.g. output/*.png)
        for p in _resolve_glob(token, script_dir):
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        # 2. repo-relative path (e.g. data/v31_*.csv, observations/output/...)
        for p in _resolve_glob(token, REPO_ROOT):
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        # 3. bare filename in data/ (legacy fallback)
        if "/" not in token:
            for p in _resolve_glob(token, REPO_ROOT / "data"):
                if p.exists():
                    mtimes.append(p.stat().st_mtime)
        # 4. archive: data/X.csv → data/archive/X_*.csv (timestamped outputs)
        for p in _resolve_archive(token):
            if p.exists():
                mtimes.append(p.stat().st_mtime)
    return max(mtimes) if mtimes else None


def _get_input_mtime(step: "PipelineStep") -> Optional[float]:
    """Return the mtime of the most-recently-modified *input* file."""
    tokens = _parse_file_tokens(step.reads)
    mtimes: List[float] = []
    for token in tokens:
        full = REPO_ROOT / token
        if full.exists():
            mtimes.append(full.stat().st_mtime)
        else:
            # archive: data/X.csv → data/archive/X_*.csv
            archive_hits = _resolve_archive(token)
            if archive_hits:
                mtimes.append(archive_hits[-1].stat().st_mtime)
    return max(mtimes) if mtimes else None


def _staleness(step: "PipelineStep"):
    """Return (last_run_at_iso, input_stale)."""
    out_mtime = _get_output_mtime(step)
    if out_mtime is None:
        return None, False   # never ran — no output yet, not stale per-se
    last_run = datetime.fromtimestamp(out_mtime).strftime("%Y-%m-%d %H:%M")
    in_mtime  = _get_input_mtime(step)
    stale = bool(in_mtime and in_mtime > out_mtime)
    return last_run, stale


def load_pipeline() -> List[PipelineStep]:
    steps: List[PipelineStep] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                order = int(row["Order"])
            except (ValueError, KeyError):
                continue
            step = PipelineStep(
                order=order,
                stage=row.get("Stage", ""),
                dependencies=row.get("Dependencies", ""),
                script=row.get("Script", ""),
                reads=row.get("Reads", ""),
                writes=row.get("Writes", ""),
                paper_section=row.get("Paper Section", ""),
            )
            step.missing_inputs = _check_inputs(step.reads, step.script)
            step.is_long = step.stage_number == "0"
            step.last_run_at, step.input_stale = _staleness(step)
            steps.append(step)
    steps.sort(key=lambda s: s.order)
    return steps


def get_output_images(step: PipelineStep) -> List[str]:
    """
    Scan output directories of the script's folder for PNG files
    written after the step ran.  Returns relative-to-repo paths.
    """
    script_dir = REPO_ROOT / Path(step.script).parent
    out_dirs = [
        script_dir / "output",
        script_dir,
    ]
    images = []
    for d in out_dirs:
        if d.is_dir():
            for p in sorted(d.glob("*.png")):
                images.append(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
    return images


def get_output_csvs(step: PipelineStep) -> List[str]:
    """Same but for CSV files produced."""
    script_dir = REPO_ROOT / Path(step.script).parent
    out_dirs = [script_dir / "output", REPO_ROOT / "data", _ARCHIVE_DIR]
    csvs = []
    for d in out_dirs:
        if d.is_dir():
            for p in sorted(d.glob("*.csv")):
                csvs.append(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
    return csvs
