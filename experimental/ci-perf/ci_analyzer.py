#!/usr/bin/env python3
"""
CI Analyzer
-----------
Transforms raw GitHub Actions workflow run data into statistics, trends,
and computed metrics for the CI highlights report.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ci_data_fetcher import CIDataFetcher, WORKFLOW_REGISTRY

logger = logging.getLogger(__name__)

MONTH_NAMES = {
    "01": "Jan",
    "02": "Feb",
    "03": "Mar",
    "04": "Apr",
    "05": "May",
    "06": "Jun",
    "07": "Jul",
    "08": "Aug",
    "09": "Sep",
    "10": "Oct",
    "11": "Nov",
    "12": "Dec",
}


def _month_label(month_key: str) -> str:
    """Convert '2026-04' to 'Apr 2026'."""
    parts = month_key.split("-")
    return f"{MONTH_NAMES.get(parts[1], parts[1])} {parts[0]}"


def _fmt_duration(minutes: Optional[float]) -> str:
    """Format minutes as human-readable string."""
    if minutes is None:
        return "N/A"
    if minutes < 1:
        return "<1m"
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins:02d}m"


def _fmt_rate(rate: Optional[float]) -> str:
    """Format a 0-1 rate as percentage string."""
    if rate is None:
        return "N/A"
    return f"{rate * 100:.0f}%"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WorkflowMonthStats:
    workflow_key: str
    workflow_label: str
    month_key: str
    month_label: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    cancelled_runs: int = 0
    success_rate: Optional[float] = None
    avg_duration_m: Optional[float] = None
    min_duration_m: Optional[float] = None
    max_duration_m: Optional[float] = None
    median_duration_m: Optional[float] = None


@dataclass
class WorkflowTrend:
    workflow_key: str
    workflow_label: str
    months: list[WorkflowMonthStats] = field(default_factory=list)
    trend_description: str = ""


@dataclass
class NightlyHealth:
    total_runs: int = 0
    successful_runs: int = 0
    success_rate: Optional[float] = None
    avg_duration_m: Optional[float] = None
    min_duration_m: Optional[float] = None
    max_duration_m: Optional[float] = None
    fastest_run_date: str = ""
    worst_run_date: str = ""
    worst_duration_m: Optional[float] = None


@dataclass
class WaitTimeEntry:
    change_area: str
    avg_wait_m: Optional[float]
    bottleneck: str


@dataclass
class ReportData:
    generated_date: str = ""
    data_period: str = ""
    month_labels: list[str] = field(default_factory=list)

    # At-a-glance
    workflow_count: int = 0
    project_count: int = 28
    avg_pr_build_m: Optional[float] = None
    nightly_pass_rate: Optional[float] = None

    # Developer wait times
    wait_times: list[WaitTimeEntry] = field(default_factory=list)

    # Nightly health
    nightly: NightlyHealth = field(default_factory=NightlyHealth)

    # Trends
    trends: list[WorkflowTrend] = field(default_factory=list)

    # Improvements and concerns (title, description)
    improvements: list[tuple[str, str]] = field(default_factory=list)
    concerns: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class CIAnalyzer:
    def __init__(
        self,
        raw_data: dict[str, dict[str, list[dict]]],
        fetcher: CIDataFetcher,
    ):
        self.raw = raw_data
        self.fetcher = fetcher
        # Ordered list of month keys (oldest first)
        sample_months = next(iter(raw_data.values()), {})
        self.month_keys = sorted(sample_months.keys())

    # ----- duration helpers -----

    @staticmethod
    def run_duration_minutes(run: dict) -> Optional[float]:
        """Compute wall-clock duration from run_started_at to updated_at."""
        started = run.get("run_started_at")
        updated = run.get("updated_at")
        if not started or not updated:
            return None
        try:
            t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            delta = (t1 - t0).total_seconds() / 60.0
            return max(0.0, delta)
        except (ValueError, TypeError):
            return None

    # ----- per-workflow stats -----

    def compute_month_stats(
        self, wf_key: str, month_key: str, runs: list[dict]
    ) -> WorkflowMonthStats:
        label = WORKFLOW_REGISTRY[wf_key]["label"]
        stats = WorkflowMonthStats(
            workflow_key=wf_key,
            workflow_label=label,
            month_key=month_key,
            month_label=_month_label(month_key),
        )
        if not runs:
            return stats

        stats.total_runs = len(runs)
        stats.successful_runs = sum(
            1 for r in runs if r.get("conclusion") == "success"
        )
        stats.failed_runs = sum(
            1 for r in runs if r.get("conclusion") == "failure"
        )
        stats.cancelled_runs = sum(
            1 for r in runs if r.get("conclusion") == "cancelled"
        )
        completed = stats.successful_runs + stats.failed_runs
        if completed > 0:
            stats.success_rate = stats.successful_runs / completed

        # Duration stats from completed runs (success + failure).
        # Exclude runs > 720m (12h) as they are CI timeout artifacts, not real durations.
        durations = []
        for r in runs:
            if r.get("conclusion") in ("success", "failure"):
                d = self.run_duration_minutes(r)
                if d is not None and 0 < d < 720:
                    durations.append(d)

        if durations:
            stats.avg_duration_m = statistics.mean(durations)
            stats.min_duration_m = min(durations)
            stats.max_duration_m = max(durations)
            stats.median_duration_m = statistics.median(durations)

        return stats

    # ----- TheRock CI classification -----

    def classify_therock_run(self, run: dict) -> Optional[str]:
        """Classify a TheRock CI run as profiler/runtimes/core/other by job names."""
        jobs = self.fetcher.get_run_jobs(run["id"])
        if not jobs:
            return None

        linux_job_names = [
            j.get("name", "")
            for j in jobs
            if "Linux" in j.get("name", "") and j.get("conclusion") != "skipped"
        ]
        if not linux_job_names:
            return "skipped"

        all_names = " ".join(linux_job_names).lower()
        if "hip-tests" in all_names or "rocrtst" in all_names:
            return "runtimes"
        if "rocprofiler" in all_names:
            return "profiler"
        return "other"

    # ----- Developer wait times -----

    def compute_wait_times(self) -> list[WaitTimeEntry]:
        """Compute average PR duration for each workflow in the most recent month."""
        if not self.month_keys:
            return []

        current_month = self.month_keys[-1]
        entries = []

        # TheRock CI - classify into profiler/runtimes
        therock_runs = self.raw.get("therock_ci", {}).get(current_month, [])
        pr_runs = [
            r
            for r in therock_runs
            if r.get("event") == "pull_request"
            and r.get("conclusion") in ("success", "failure")
        ]

        # Classify a sample of PR runs
        profiler_durations = []
        runtimes_durations = []
        sample = [
            r
            for r in pr_runs
            if (self.run_duration_minutes(r) or 0) > 10
        ][:15]

        for run in sample:
            d = self.run_duration_minutes(run)
            if d is None:
                continue
            classification = self.classify_therock_run(run)
            if classification == "profiler":
                profiler_durations.append(d)
            elif classification == "runtimes":
                runtimes_durations.append(d)

        if profiler_durations:
            entries.append(
                WaitTimeEntry(
                    "Profiler (rocprofiler-*)",
                    statistics.mean(profiler_durations),
                    "TheRock build",
                )
            )
        if runtimes_durations:
            entries.append(
                WaitTimeEntry(
                    "Runtimes (hip, clr, rocr)",
                    statistics.mean(runtimes_durations),
                    "TheRock build",
                )
            )

        # Per-project pipelines
        per_project = [
            ("amdsmi", "AMDSMI", "Parallel distro builds"),
            ("media_libs", "Media libs (rocdecode/rocjpeg)", "Dedicated pipeline"),
            ("aqlprofile", "AqlProfile", "Multi-GPU matrix"),
            ("rccl", "RCCL", "Build + GPU test"),
            ("rocprof_systems", "rocprofiler-systems", "Per-distro builds"),
            ("rocprof_sdk", "rocprofiler-sdk", "Multi-distro + sanitizers"),
        ]
        for wf_key, area_name, bottleneck in per_project:
            runs = self.raw.get(wf_key, {}).get(current_month, [])
            completed = [
                r
                for r in runs
                if r.get("conclusion") in ("success", "failure")
                and r.get("event") in ("pull_request", "push")
            ]
            durations = []
            for r in completed:
                d = self.run_duration_minutes(r)
                if d is not None and 0 < d < 720:
                    durations.append(d)
            if durations:
                entries.append(
                    WaitTimeEntry(area_name, statistics.mean(durations), bottleneck)
                )

        # Sort by duration descending
        entries.sort(key=lambda e: e.avg_wait_m or 0, reverse=True)
        return entries

    # ----- Nightly health -----

    def compute_nightly_health(self, recent_n: int = 20) -> NightlyHealth:
        """Compute nightly health from the most recent N runs across all months."""
        health = NightlyHealth()
        all_nightly_runs = []
        for month_key in self.month_keys:
            all_nightly_runs.extend(
                self.raw.get("therock_nightly", {}).get(month_key, [])
            )

        if not all_nightly_runs:
            return health

        # Sort by created_at descending, take recent_n
        all_nightly_runs.sort(
            key=lambda r: r.get("created_at", ""), reverse=True
        )
        recent = all_nightly_runs[:recent_n]

        health.total_runs = len(recent)
        health.successful_runs = sum(
            1 for r in recent if r.get("conclusion") == "success"
        )
        completed = sum(
            1
            for r in recent
            if r.get("conclusion") in ("success", "failure")
        )
        if completed > 0:
            health.success_rate = health.successful_runs / completed

        durations = []
        for r in recent:
            d = self.run_duration_minutes(r)
            if d is not None and d > 0:
                durations.append((d, r))

        if durations:
            health.avg_duration_m = statistics.mean([d for d, _ in durations])
            min_d, min_r = min(durations, key=lambda x: x[0])
            max_d, max_r = max(durations, key=lambda x: x[0])
            health.min_duration_m = min_d
            health.max_duration_m = max_d
            health.fastest_run_date = (
                min_r.get("created_at", "")[:10]
            )
            health.worst_run_date = max_r.get("created_at", "")[:10]
            health.worst_duration_m = max_d

        return health

    # ----- Trends -----

    def compute_trends(self) -> list[WorkflowTrend]:
        """Compute per-workflow trends across all months."""
        trends = []
        for wf_key, wf_info in WORKFLOW_REGISTRY.items():
            trend = WorkflowTrend(
                workflow_key=wf_key, workflow_label=wf_info["label"]
            )
            for month_key in self.month_keys:
                runs = self.raw.get(wf_key, {}).get(month_key, [])
                stats = self.compute_month_stats(wf_key, month_key, runs)
                trend.months.append(stats)

            # Compute trend description
            trend.trend_description = self._describe_trend(trend)
            trends.append(trend)
        return trends

    @staticmethod
    def _describe_trend(trend: WorkflowTrend) -> str:
        """Generate a human-readable trend description."""
        months_with_data = [
            m for m in trend.months if m.avg_duration_m is not None
        ]
        if len(months_with_data) < 2:
            if not months_with_data:
                return "No data"
            return "Insufficient history"

        oldest = months_with_data[0]
        newest = months_with_data[-1]

        if oldest.avg_duration_m and newest.avg_duration_m:
            delta_pct = (
                (newest.avg_duration_m - oldest.avg_duration_m)
                / oldest.avg_duration_m
                * 100
            )
            if abs(delta_pct) < 10:
                return "Stable"
            elif delta_pct > 0:
                return f"+{delta_pct:.0f}% growth"
            else:
                return f"{delta_pct:.0f}% (faster)"

        # Check for newly-created workflow
        no_data_months = [
            m for m in trend.months if m.total_runs == 0
        ]
        if no_data_months and months_with_data:
            return f"Created in {months_with_data[0].month_label}"

        return ""

    # ----- Improvements / Concerns -----

    def compute_changes(
        self, trends: list[WorkflowTrend]
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Identify improvements and concerns from trend data."""
        improvements = []
        concerns = []

        for trend in trends:
            if len(trend.months) < 2:
                continue

            # Compare most recent two months with data
            months_with_data = [
                m for m in trend.months if m.avg_duration_m is not None
            ]
            if len(months_with_data) >= 2:
                prev = months_with_data[-2]
                curr = months_with_data[-1]

                if prev.avg_duration_m and curr.avg_duration_m:
                    delta_pct = (
                        (curr.avg_duration_m - prev.avg_duration_m)
                        / prev.avg_duration_m
                        * 100
                    )
                    prev_str = _fmt_duration(prev.avg_duration_m)
                    curr_str = _fmt_duration(curr.avg_duration_m)
                    label = trend.workflow_label

                    if delta_pct < -15:
                        improvements.append((
                            f"{label}: {abs(delta_pct):.0f}% Faster",
                            f"{prev_str} ({prev.month_label}) -> "
                            f"{curr_str} ({curr.month_label}).",
                        ))
                    elif delta_pct > 15:
                        concerns.append((
                            f"{label}: +{delta_pct:.0f}% Slower "
                            f"({prev_str} -> {curr_str})",
                            f"Duration increased from {prev.month_label} "
                            f"to {curr.month_label}.",
                        ))

                # Success rate changes
                if prev.success_rate is not None and curr.success_rate is not None:
                    rate_delta = (curr.success_rate - prev.success_rate) * 100
                    if rate_delta < -10:
                        concerns.append((
                            f"{trend.workflow_label}: Success Rate "
                            f"{_fmt_rate(prev.success_rate)} -> "
                            f"{_fmt_rate(curr.success_rate)}",
                            f"Pass rate dropped {abs(rate_delta):.0f} "
                            f"percentage points.",
                        ))

            # Previously broken (no successes) now working
            all_months = trend.months
            if len(all_months) >= 2:
                older = all_months[:-1]
                newest = all_months[-1]
                older_had_success = any(m.successful_runs > 0 for m in older)
                older_had_runs = any(m.total_runs > 0 for m in older)
                if (
                    not older_had_success
                    and older_had_runs
                    and newest.successful_runs > 0
                ):
                    improvements.append((
                        f"{trend.workflow_label}: Now Working",
                        f"Previously had no successful runs; "
                        f"now {newest.successful_runs}/{newest.total_runs} "
                        f"passing in {newest.month_label}.",
                    ))

            # Newly created workflow
            if all_months and all_months[0].total_runs == 0:
                first_active = next(
                    (m for m in all_months if m.total_runs > 0), None
                )
                if first_active:
                    improvements.append((
                        f"{trend.workflow_label}: New Pipeline",
                        f"Created in {first_active.month_label} with "
                        f"{first_active.total_runs} runs.",
                    ))

        return improvements, concerns

    # ----- Main assembly -----

    def build_report_data(self) -> ReportData:
        """Assemble all computed metrics into a ReportData object."""
        data = ReportData()
        data.generated_date = datetime.now().strftime("%B %d, %Y")
        data.month_labels = [_month_label(mk) for mk in self.month_keys]

        if self.month_keys:
            first = _month_label(self.month_keys[0])
            last = _month_label(self.month_keys[-1])
            data.data_period = f"{first} - {last}"

        # Count workflows
        data.workflow_count = len(
            self.fetcher._workflow_id_cache
        ) or len(WORKFLOW_REGISTRY)

        # Wait times (also gives us avg PR build time)
        logger.info("Computing developer wait times...")
        data.wait_times = self.compute_wait_times()

        # Avg PR build time from TheRock CI profiler path
        profiler_entry = next(
            (w for w in data.wait_times if "Profiler" in w.change_area), None
        )
        data.avg_pr_build_m = profiler_entry.avg_wait_m if profiler_entry else None

        # Nightly health
        logger.info("Computing nightly health...")
        data.nightly = self.compute_nightly_health()
        data.nightly_pass_rate = data.nightly.success_rate

        # Trends
        logger.info("Computing trends...")
        data.trends = self.compute_trends()

        # Improvements / Concerns
        logger.info("Computing improvements and concerns...")
        data.improvements, data.concerns = self.compute_changes(data.trends)

        return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fetcher = CIDataFetcher()
    raw = fetcher.fetch_all(months=3)
    analyzer = CIAnalyzer(raw, fetcher)
    report = analyzer.build_report_data()

    print(f"\nReport: {report.data_period}")
    print(f"Workflows tracked: {report.workflow_count}")
    print(f"Avg PR build: {_fmt_duration(report.avg_pr_build_m)}")
    print(f"Nightly pass rate: {_fmt_rate(report.nightly_pass_rate)}")
    print(f"\nWait times:")
    for w in report.wait_times:
        print(f"  {w.change_area}: {_fmt_duration(w.avg_wait_m)} ({w.bottleneck})")
    print(f"\nTrends:")
    for t in report.trends:
        vals = " -> ".join(
            _fmt_duration(m.avg_duration_m) for m in t.months
        )
        print(f"  {t.workflow_label}: {vals} ({t.trend_description})")
    print(f"\nImprovements: {len(report.improvements)}")
    for title, desc in report.improvements:
        print(f"  + {title}: {desc}")
    print(f"\nConcerns: {len(report.concerns)}")
    for title, desc in report.concerns:
        print(f"  ! {title}: {desc}")
