"""
Prediction helper functions – V2 (append-only with existing-weekend checks)

Exports SSOT-style classification CSVs for sessions that have already happened,
**appending only missing weekends** to per-session CSVs (no re-adding past events).

Output folder:
    data/predictions/ssot/

Output files (maintained per season):
    {season}_qualifying.csv
    {season}_race.csv
    {season}_sprint_qualifying.csv
    {season}_sprint.csv

Notes:
    * Session inclusion is driven by the official schedule and the same
      "completed/has started by now" logic used elsewhere:
      `_official_schedule(...)` + `_sessions_completed(...)`.
    * Each file contains all completed events for the specified `season` for that session type.
    * A session is included only if `sess.results` is present and non-empty.
    * Idempotent: re-running after a weekend will append only the new weekend’s rows.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Dict, List, Optional, Set, Tuple
from collections.abc import Iterable

import pandas as _pd

from .general_utils import _official_schedule, _sessions_completed

# -----------------------------------------------------------------------------
# Logging / warnings
# -----------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*dtype incompatible with datetime64\\[ns\\].*",
    module="fastf1",
)
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
_logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
SSOT_DIR = "data/predictions/ssot"

# FastF1 canonical names → filename stems
SESSION_TO_STEM = {
    "Sprint Qualifying": "sprint_qualifying",
    "Sprint Shootout":   "sprint_qualifying",  # legacy alias
    "Sprint":            "sprint",
    "Qualifying":        "qualifying",
    "Race":              "race",
}


# Schedule helper labels → canonical FastF1 names
LABEL_TO_SESSION_NAME = {
    "Q":  "Qualifying",
    "R":  "Race",
    "S":  "Sprint",
    "SQ": "Sprint Qualifying",
    "SS": "Sprint Shootout",
}

# Stable subset (we preserve extra columns after these)
DEFAULT_KEEP_COLS = [
    # meta
    "WeekendId", "Season", "RoundNumber", "EventName", "SessionName", "SessionStart",
    # driver/team
    "DriverNumber", "Abbreviation", "DriverId", "BroadcastName", "TeamName",
    # classification
    "GridPosition", "ClassifiedPosition", "Status",
    # quali timing (when present)
    "Q1", "Q2", "Q3",
    # bests (when present)
    "BestLapTime", "BestLapSpeed",
]

# Per-session CSV identity key (per file we store only one session type)
DEDUP_KEYS = ["WeekendId", "DriverNumber"]

# Treat "Sprint Shootout" as the same bucket/file as "Sprint Qualifying"
BUCKET_ALIAS = {
    "Sprint Shootout": "Sprint Qualifying",
}

def _bucket_session_name(name: str) -> str:
    """Map session names to their storage bucket (file)."""
    return BUCKET_ALIAS.get(name, name)

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExportResult:
    """Result of one export/append operation.

    Attributes:
        session_name: Canonical FastF1 session name (e.g., "Qualifying", "Race").
        written_path: Filesystem path written/modified for this export, if any.
        status: One of {"written", "appended", "skipped", "error"}.
        message: Extra context (e.g., "No new weekends to append." or error info).
    """
    session_name: str
    written_path: Optional[str]
    status: str
    message: Optional[str] = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    """Ensure a directory exists (create parents as needed).

    Args:
        path: Directory path to create if missing.

    Side Effects:
        Creates directories on disk when they do not exist.
    """
    _Path(path).mkdir(parents=True, exist_ok=True)


def _to_csv(df: _pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV, ensuring parent directory exists.

    Args:
        df: Data to write.
        path: Output CSV path.

    Side Effects:
        Writes a CSV file to disk.
    """
    _ensure_dir(str(_Path(path).parent))
    df.to_csv(path, index=False)


def _results_ready(sess) -> bool:
    """Check if a FastF1 session has completed results available.

    Args:
        sess: Loaded FastF1 session.

    Returns:
        True if `sess.results` exists and is non-empty; False otherwise.
    """
    try:
        df = sess.results
        return (df is not None) and (len(df) > 0)
    except Exception:
        return False


def _event_get(ev, *keys, default=None):
    """Safely fetch an attribute or mapping key from a FastF1 Event.

    Tries multiple keys to be robust to version/casing differences.

    Args:
        ev: Event-like object returned by FastF1 (or mapping-like).
        *keys: Candidate attribute names or mapping keys to try, in order.
        default: Fallback value if none of the keys are found.

    Returns:
        The first successfully retrieved value, or `default`.
    """
    for k in keys:
        if hasattr(ev, k):
            try:
                return getattr(ev, k)
            except Exception:
                pass
        try:
            if ev is not None and hasattr(ev, "__getitem__"):
                return ev[k]  # may raise
        except Exception:
            continue
    return default


def _build_results_df(sess, session_name: str) -> _pd.DataFrame:
    """Normalize FastF1 session results to a stable SSOT schema.

    Args:
        sess: Loaded session object (must have `results` populated).
        session_name: Canonical session name (e.g., "Qualifying", "Race", "Sprint").

    Returns:
        Normalized results with a stable set of meta and classification columns.
        Guaranteed columns:
            WeekendId, Season, RoundNumber, EventName, SessionName, SessionStart,
            DriverNumber, Abbreviation, DriverId, BroadcastName, TeamName,
            GridPosition, ClassifiedPosition, Status, Q1, Q2, Q3, BestLapTime, BestLapSpeed.

    Notes:
        * Handles differences in FastF1 event attribute casing (e.g., year vs Year).
        * Preserves any extra columns present in `sess.results` after the standard ones.
    """
    df = sess.results.copy()

    # Robust event meta
    year_val  = _event_get(sess.event, "Year", "year")
    round_val = _event_get(sess.event, "RoundNumber", "round", "Round")
    event_nm  = _event_get(sess.event, "EventName", "OfficialEventName", "name", "Name")

    weekend_id = f"{int(year_val)}_{int(round_val):02d}"
    df.insert(0, "WeekendId", weekend_id)
    df.insert(1, "Season", int(year_val))
    df.insert(2, "RoundNumber", int(round_val))
    df.insert(3, "EventName", event_nm)
    df.insert(4, "SessionName", session_name)

    # Session start (ISO-UTC when available)
    start_attr = getattr(sess, "session_start_time", None)
    if isinstance(start_attr, _pd.Timestamp):
        session_start = start_attr.tz_convert("UTC").isoformat()
    elif start_attr is not None:
        session_start = str(start_attr)
    else:
        session_start = None
    df.insert(5, "SessionStart", session_start)

    # Ensure stable columns exist
    for col in DEFAULT_KEEP_COLS:
        if col not in df.columns:
            df[col] = _pd.NA

    # Order: standard subset first, then extras
    keep = [c for c in DEFAULT_KEEP_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in set(keep)]
    return df[keep + rest].copy()


def _collect_results_for_event(
    season: int,
    gp_name: str,
    session_names: List[str],
) -> Dict[str, _pd.DataFrame]:
    """Load completed session results for a given Grand Prix and normalize them.

    Args:
        season: Championship year.
        gp_name: Grand Prix name (must match FastF1 naming, e.g., "Italian Grand Prix").
        session_names: Canonical session names to attempt (subset of:
            "Sprint Qualifying", "Sprint Shootout", "Sprint", "Qualifying", "Race").

    Returns:
        Map of session name → normalized results DataFrame for sessions where
        results are available. Sessions without results are omitted.

    Raises:
        RuntimeError: If `fastf1` is not available in the environment.

    Notes:
        Conservative by design: sessions are included only if `sess.results`
        is present and non-empty.
    """
    out: Dict[str, _pd.DataFrame] = {}
    try:
        import fastf1 as _ff1
    except Exception as exc:
        raise RuntimeError("fastf1 not available in this environment.") from exc

    for sname in session_names:
        try:
            sess = _ff1.get_session(season, gp_name, sname)
            sess.load()
        except Exception:
            # Non-sprint weekend or cache/network hiccup — skip silently.
            continue

        if not _results_ready(sess):
            continue

        out[sname] = _build_results_df(sess, sname)

    return out


def _align_union_columns(existing: _pd.DataFrame, incoming: _pd.DataFrame) -> Tuple[_pd.DataFrame, _pd.DataFrame]:
    """Column-union align two DataFrames so they can be concatenated safely.

    Args:
        existing: Previously stored DataFrame (from disk).
        incoming: New rows to append.

    Returns:
        A tuple of (existing_aligned, incoming_aligned) with the same ordered columns.
    """
    all_cols = list(dict.fromkeys(list(existing.columns) + list(incoming.columns)))
    return existing.reindex(columns=all_cols), incoming.reindex(columns=all_cols)


def _coerce_dedup_key_types(df: _pd.DataFrame) -> _pd.DataFrame:
    """Cast dedup keys to string to avoid '1' vs '01' / int vs str mismatches.

    Args:
        df: DataFrame to coerce in place for dedup keys.

    Returns:
        The same DataFrame, with DEDUP_KEYS cast to string when present.
    """
    for k in DEDUP_KEYS:
        if k in df.columns:
            df[k] = df[k].astype(str)
    return df


def _read_existing_weekend_ids(season: int, sname: str) -> Set[str]:
    """Load existing season CSV for this session and return its WeekendIds.

    Args:
        season: Championship year.
        sname: Canonical session name.

    Returns:
        A set of WeekendId strings already stored for this session. If the season
        CSV does not yet exist, returns an empty set.
    """
    sname_bucket = _bucket_session_name(sname)
    path = f"{SSOT_DIR}/{season}_{SESSION_TO_STEM[sname_bucket]}.csv"
    if not _Path(path).exists():
        return set()
    try:
        df = _pd.read_csv(path, usecols=["WeekendId"])
        return set(df["WeekendId"].astype(str).unique())
    except Exception:
        return set()


def _append_dedup_by_keys(path: str, new_parts: List[_pd.DataFrame]) -> ExportResult:
    """Append new parts into `path` with unioned columns and dedup by DEDUP_KEYS.

    Args:
        path: Output CSV path for a given session type and season.
        new_parts: List of normalized DataFrames to append.

    Returns:
        ExportResult with status "written" (created), "appended" (added rows), or "skipped".
    """
    if not new_parts:
        return ExportResult("UNKNOWN", None, "skipped", "No new parts to append.")

    df_new = _pd.concat(new_parts, ignore_index=True)
    df_new = _coerce_dedup_key_types(df_new)

    if "RoundNumber" in df_new.columns:
        df_new = df_new.sort_values(["RoundNumber", "DriverNumber"], kind="mergesort")

    if _Path(path).exists():
        df_old = _pd.read_csv(path)
        df_old = _coerce_dedup_key_types(df_old)
        df_old, df_new = _align_union_columns(df_old, df_new)
        combined = _pd.concat([df_old, df_new], ignore_index=True)

        if all(k in combined.columns for k in DEDUP_KEYS):
            combined = combined.drop_duplicates(DEDUP_KEYS, keep="first")  # keep previously stored rows

        combined.to_csv(path, index=False)
        return ExportResult("UNKNOWN", path, "appended", "Appended new weekends (deduped).")
    else:
        _to_csv(df_new, path)
        return ExportResult("UNKNOWN", path, "written", "Created file with first data batch.")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def export_completed_classifications_csv(
    season: int,
    include_sprint: bool = True,
    up_to_utc: "datetime | None" = None,
) -> Dict[str, ExportResult]:
    """Append only **missing** weekends to per-session season CSVs.

    Args:
        season: Championship year to export.
        include_sprint: If True, also maintain Sprint Qualifying/Shootout and Sprint files.
        up_to_utc: Optional UTC cutoff timestamp. Only sessions that have *started*
            before this instant (per the official schedule) are considered. If None, uses now.

    Returns:
        Map of canonical session name → ExportResult. Status is one of
        {"written", "appended", "skipped", "error"}.

    Behavior:
        * Uses `_official_schedule` + `_sessions_completed` to decide which sessions should exist by `up_to_utc`.
        * Loads FastF1 `sess.results` for those sessions, normalizes columns, and **appends** to the season CSV for that session type.
        * **Skips** weekends already present by checking existing `WeekendId`s up front.
        * Deduplicates by (WeekendId, DriverNumber) as a final guard.

    Examples:
        >>> # Export/append all completed sessions up to now
        >>> export_completed_classifications_csv(2025, include_sprint=True)
    """
    from datetime import datetime, timezone

    sched = _official_schedule(season)
    now = up_to_utc or datetime.now(timezone.utc)

    wanted_codes = {"Q", "R"}
    if include_sprint:
        wanted_codes |= {"SQ", "SS", "S"}

    # 1) Snapshot which weekends are already stored per session
    existing_ids: Dict[str, Set[str]] = {
        "Qualifying": _read_existing_weekend_ids(season, "Qualifying"),
        "Race": _read_existing_weekend_ids(season, "Race"),
        "Sprint Qualifying": _read_existing_weekend_ids(season, "Sprint Qualifying"),
        "Sprint": _read_existing_weekend_ids(season, "Sprint"),
    }

    # 2) Collect only weekends NOT already present
    acc_new: Dict[str, List[_pd.DataFrame]] = {
        "Qualifying": [],
        "Race": [],
        "Sprint Qualifying": [],
        "Sprint": [],
    }

    for _, ev in sched.iterrows():
        fmt = (ev.EventFormat or "").strip().lower()
        name = ev.EventName or ""
        fp1_utc = ev.Session1DateUtc

        # Skip pre-season testing
        if fmt == "testing" or "test" in name.lower():
            continue

        completed = _sessions_completed(fmt, fp1_utc, now)
        to_try = [LABEL_TO_SESSION_NAME[c] for c in completed if c in wanted_codes]

        # Collect normalized results if available
        res_map = _collect_results_for_event(season, name, to_try)
        for sname, df in res_map.items():
            bucket = _bucket_session_name(sname)
            df = _coerce_dedup_key_types(df)
            wk_ids = set(df["WeekendId"].astype(str).unique())
            missing = [wid for wid in wk_ids if wid not in existing_ids[bucket]]
            if not missing:
                continue

            df_new_only = df[df["WeekendId"].astype(str).isin(missing)]
            if not df_new_only.empty:
                acc_new[bucket].append(df_new_only)
                existing_ids[bucket].update(missing)

    # 3) Append new rows per session file
    _ensure_dir(SSOT_DIR)
    results: Dict[str, ExportResult] = {}

    for sname, parts in acc_new.items():
        stem = SESSION_TO_STEM[sname]
        outpath = f"{SSOT_DIR}/{season}_{stem}.csv"

        if not parts:
            if _Path(outpath).exists():
                results[sname] = ExportResult(sname, outpath, "skipped", "No new weekends to append.")
            else:
                results[sname] = ExportResult(sname, None, "skipped", "Nothing to write yet.")
            continue

        res = _append_dedup_by_keys(outpath, parts)
        # Normalize name in result
        results[sname] = ExportResult(sname, res.written_path, res.status, res.message)
        _logger.info("[%s] %s → %s", sname, res.status, res.written_path or res.message)

    return results

def export_completed_classifications_csv_multi(
    seasons: Iterable[int],
    include_sprint: bool = True,
    up_to_utc: "datetime | None" = None,
) -> Dict[int, Dict[str, ExportResult]]:
    """Run the append-only export across multiple seasons.

    Args:
        seasons: Iterable of championship years (e.g., `range(2023, 2026)` or `[2023, 2024, 2025]`).
        include_sprint: If True, also maintain Sprint Qualifying/Shootout and Sprint files.
        up_to_utc: Optional UTC cutoff timestamp. Only sessions that have *started*
            before this instant (per the official schedule) are considered. If None, uses now.

    Returns:
        Dict mapping `season -> {session_name -> ExportResult}`.

    Notes:
        * Each season writes/updates its own per-session CSVs:
          `{season}_qualifying.csv`, `{season}_race.csv`, `{season}_sprint_qualifying.csv`, `{season}_sprint.csv`.
        * Idempotent: per season we only append **missing** weekends.
    """
    results: Dict[int, Dict[str, ExportResult]] = {}
    for yr in seasons:
        results[yr] = export_completed_classifications_csv(
            season=yr,
            include_sprint=include_sprint,
            up_to_utc=up_to_utc,
        )
    return results


def export_completed_classifications_csv_range(
    start_year: int,
    end_year: int,
    include_sprint: bool = True,
    up_to_utc: "datetime | None" = None,
) -> Dict[int, Dict[str, ExportResult]]:
    """Run the append-only export for an inclusive year range.

    Args:
        start_year: First season (inclusive).
        end_year: Last season (inclusive). Must be >= start_year.
        include_sprint: If True, also maintain Sprint Qualifying/Shootout and Sprint files.
        up_to_utc: Optional UTC cutoff timestamp applied to each season.

    Returns:
        Dict mapping `season -> {session_name -> ExportResult}`.

    Raises:
        ValueError: If `end_year < start_year`.

    Examples:
        >>> # Append-only export for 2023–2025
        >>> export_completed_classifications_csv_range(2023, 2025)
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year})")
    seasons = range(start_year, end_year + 1)
    return export_completed_classifications_csv_multi(
        seasons=seasons,
        include_sprint=include_sprint,
        up_to_utc=up_to_utc,
    )
