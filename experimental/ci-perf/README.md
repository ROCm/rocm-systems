# CI Performance Reports

Tools for generating PDF reports from GitHub Actions CI data for the rocm-systems repository.

## Reports

| Script | Description | Data Source |
|--------|-------------|-------------|
| `generate_ci_highlights.py` | Executive summary (7 pages) with key metrics, trends, and alerts | Live GitHub Actions API |
| `generate_ci_report.py` | Full analysis (24 pages) with detailed per-workflow breakdowns | Hardcoded (snapshot) |

## Prerequisites

- Python 3.10+
- GitHub token: set `GH_TOKEN` env var, or authenticate with `gh auth login`

## Setup

```bash
cd experimental/ci-perf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Highlights Report (dynamic)

Fetches live data from the GitHub Actions API and generates a summary PDF.

```bash
# Default: last 3 months, outputs rocm-systems-ci-highlights.pdf
python generate_ci_highlights.py

# Custom options
python generate_ci_highlights.py --repo ROCm/rocm-systems --months 2 --output my-report.pdf
```

**CLI flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--repo` | `ROCm/rocm-systems` | GitHub repository (owner/repo) |
| `--months` | `3` | Number of months to analyze |
| `--output` | `rocm-systems-ci-highlights.pdf` | Output PDF filename |

### Full Analysis Report (static)

Generates the detailed report from hardcoded data.

```bash
python generate_ci_report.py
```

## Architecture

```
ci_data_fetcher.py      GitHub Actions API client (auth, retry, pagination)
        |
        v
ci_analyzer.py          Statistics engine (durations, trends, alerts)
        |
        v
generate_ci_highlights.py   PDF renderer (tables, charts, formatting)
```

### Tracked Workflows

| Workflow | File |
|----------|------|
| TheRock CI | `therock-ci.yml` |
| TheRock CI Nightly | `therock-ci-nightly.yml` |
| AMDSMI CI | `amdsmi-build.yml` |
| Media Libs CI | `media-libs-ci.yml` |
| AqlProfile CI | `aqlprofile-continuous_integration.yml` |
| RCCL CI | `therock-rccl-ci.yml` |
| rocprofiler-systems CI | `rocprofiler-systems-continuous-integration.yml` |
| rocprofiler-sdk CI | `rocprofiler-sdk-continuous_integration.yml` |
