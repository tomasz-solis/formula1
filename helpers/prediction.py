"""
Prediction helper functions - V1

Exports clean SSOT-style classification CSVs for sessions that have already happened,
with one CSV per session type (Qualifying, Race, Sprint Qualifying/Shootout, Sprint).

Output folder:
    data/predictions/ssot/

Output files (written only if data exists up to the cutoff time):
    {season}_qualifying.csv
    {season}_race.csv
    {season}_sprint_qualifying.csv
    {season}_sprint.csv

Notes:
    - Session inclusion is driven by the official schedule and the same
      "completed/has started by now" logic used elsewhere:
      `_official_schedule(...)` + `_sessions_completed(...)`.
    - Each file contains all completed events for the specified `season` for that session type.
    - Results are gated by FastF1 availability: a session is written only if `sess.results` is present.
    - Safe to call repeatedly: by default, does not overwrite existing CSVs unless `overwrite=True`.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Dict, List, Optional

import pandas as _pd

from .general_utils import _official_schedule, _sessions_completed

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*dtype incompatible with datetime64\\[ns\\].*",
    module="fastf1",
)
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
_logger = logging.getLogger(__name__)

SSOT_DIR = "data/predictions/ssot"

# FastF1 canonical names -> filename stems
SESSION_TO_STEM = {
    "Sprint Qualifying": "sprint_qualifying",
    "Sprint Shootout":   "sprint_qualifying",
    "Sprint":            "sprint",
    "Qualifying":        "qualifying",
    "Race":              "race",
}

# Schedule helper labels -> FastF1 canonical names
LABEL_TO_SESSION_NAME = {
    "Q":  "Qualifying",
    "R":  "Race",
    "S":  "Sprint",
    "SQ": "Sprint Qualifying",
    "SS": "Sprint Shootout",
}

DEFAULT_KEEP_COLS = [
    "WeekendId", "Season", "RoundNumber", "EventName", "SessionName", "SessionStart",
    "DriverNumber", "Abbreviation", "DriverId", "BroadcastName", "TeamName",
    "GridPosition", "ClassifiedPosition", "Status",
    "Q1", "Q2", "Q3",
    "BestLapTime", "BestLapSpeed",
]


@dataclass(frozen=True)
class ExportResult:
    """Result of one export operation.

    Attributes:
        session_name: Canonical FastF1 session name, e.g. "Qualifying", "Race".
        written_path: Filesystem path written for this export, if any.
        status: One of {"written", "skipped", "error"} describing the outcome.
        message: Extra context (e.g., "File exists" or an error description).
    """
    session_name: str
    written_path: Optional[str]
    status: str
    message: Optional[str] = None


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
    """Safely fetch an attribute or mapping key from a FastF1 Event across versions/casings.

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
                return ev[k]
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
        Columns guaranteed to exist include:
        WeekendId, Season, RoundNumber, EventName, SessionName, SessionStart,
        DriverNumber, Abbreviation, DriverId, BroadcastName, TeamName,
        GridPosition, ClassifiedPosition, Status, Q1, Q2, Q3, BestLapTime, BestLapSpeed.

    Notes:
        - Handles differences in FastF1 event attribute casing (e.g., year vs Year).
        - Preserves any extra columns present in `sess.results` after the standard ones.
    """
    df = sess.results.copy()

    year_val  = _event_get(sess.event, "Year", "year")
    round_val = _event_get(sess.event, "RoundNumber", "round", "Round")
    event_nm  = _event_get(sess.event, "EventName", "OfficialEventName", "name", "Name")

    weekend_id = f"{int(year_val)}_{int(round_val):02d}"
    df.insert(0, "WeekendId", weekend_id)
    df.insert(1, "Season", int(year_val))
    df.insert(2, "RoundNumber", int(round_val))
    df.insert(3, "EventName", event_nm)
    df.insert(4, "SessionName", session_name)

    start_attr = getattr(sess, "session_start_time", None)
    if isinstance(start_attr, _pd.Timestamp):
        session_start = start_attr.tz_convert("UTC").isoformat()
    elif start_attr is not None:
        session_start = str(start_attr)
    else:
        session_start = None
    df.insert(5, "SessionStart", session_start)

    for col in DEFAULT_KEEP_COLS:
        if col not in df.columns:
            df[col] = _pd.NA

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
        session_names: Canonical FastF1 session names to attempt (subset of:
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


def export_completed_classifications_csv(
    season: int,
    include_sprint: bool = True,
    overwrite: bool = False,
    up_to_utc: "datetime | None" = None,
) -> Dict[str, ExportResult]:
    """Export ONE CSV per session type for all sessions that have already happened.

    Args:
        season: Championship year to export.
        include_sprint: If True, also export Sprint Qualifying/Shootout and Sprint files.
        overwrite: If False, skip writing when the target CSV already exists;
            if True, overwrite existing CSVs.
        up_to_utc: Optional UTC cutoff timestamp. Only sessions that have *started*
            before this instant (per the official schedule) are considered. If None, uses now.

    Returns:
        Map of session type → ExportResult with status/written_path information.
        Keys are the canonical session names: "Qualifying", "Race", "Sprint Qualifying", "Sprint".

    Side Effects:
        Writes up to four CSV files into `data/predictions/ssot/`:
            - {season}_qualifying.csv
            - {season}_race.csv
            - {season}_sprint_qualifying.csv
            - {season}_sprint.csv

    Notes:
        - Session inclusion is derived from `_official_schedule(...)` and `_sessions_completed(...)`
          so it respects sprint vs non-sprint formats.
        - Each per-session CSV concatenates rows across all completed events in ascending RoundNumber order.
        - Results are included only if FastF1 exposes `sess.results` for the session.

    Examples:
        >>> from datetime import datetime, timezone
        >>> # Export everything up to "now"
        >>> export_completed_classifications_csv(2025, include_sprint=True, overwrite=False)
        {'Qualifying': ExportResult(...), 'Race': ExportResult(...), ...}
        >>> # Export only sessions that started before a cutoff
        >>> cutoff = datetime(2025, 9, 6, 12, 0, tzinfo=timezone.utc)
        >>> export_completed_classifications_csv(2025, include_sprint=True, up_to_utc=cutoff)
    """
    from datetime import datetime, timezone

    sched = _official_schedule(season)
    now = up_to_utc or datetime.now(timezone.utc)

    wanted_codes = {"Q", "R"}
    if include_sprint:
        wanted_codes |= {"SQ", "SS", "S"}

    acc: Dict[str, List[_pd.DataFrame]] = {
        "Qualifying": [],
        "Race": [],
        "Sprint Qualifying": [],
        "Sprint": [],
    }

    for _, ev in sched.iterrows():
        fmt = (ev.EventFormat or "").strip().lower()
        name = ev.EventName or ""
        fp1_utc = ev.Session1DateUtc

        if fmt == "testing" or "test" in name.lower():
            continue

        completed = _sessions_completed(fmt, fp1_utc, now)
        to_try = [LABEL_TO_SESSION_NAME[c] for c in completed if c in wanted_codes]

        res_map = _collect_results_for_event(season, name, to_try)
        for sname, df in res_map.items():
            acc[sname].append(df)

    _ensure_dir(SSOT_DIR)
    written: Dict[str, ExportResult] = {}

    for sname, parts in acc.items():
        if not parts:
            written[sname] = ExportResult(sname, None, "skipped", "No completed sessions or results yet.")
            continue

        df = _pd.concat(parts, ignore_index=True)
        if "RoundNumber" in df.columns:
            df = df.sort_values(["RoundNumber", "DriverNumber"], kind="mergesort")

        stem = SESSION_TO_STEM[sname]
        outpath = f"{SSOT_DIR}/{season}_{stem}.csv"

        if _Path(outpath).exists() and not overwrite:
            written[sname] = ExportResult(sname, outpath, "skipped", "File exists; use overwrite=True to replace.")
            continue

        _to_csv(df, outpath)
        written[sname] = ExportResult(sname, outpath, "written", None)
        _logger.info("Wrote %s rows to %s", len(df), outpath)

    return written
