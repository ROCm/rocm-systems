# Mainline Coverage Tracking

This directory contains coverage reports for the `amd-mainline` branch, used for project management and quality tracking.

## Files

- **`mainline-coverage-latest.xml`** - Always contains the most recent mainline coverage
- **`mainline-coverage-YYYYMMDD_HHMMSS.xml`** - Timestamped archives of previous coverage reports

## Automated Updates

Coverage is automatically updated via GitHub Actions workflow when:
- Code is pushed to `amd-mainline` branch
- Staging is promoted to mainline

## Manual Updates

To manually update coverage (if needed):

```bash
git checkout amd-mainline
./utils/update_mainline_coverage.sh
```

## CDash Integration

Coverage reports are also uploaded to CDash for dashboard visibility:
- Project: rocprofiler-compute
- Build Name Format: `mainline-{git-sha}-{date}`
- Mode: Nightly

## Coverage Interpretation

The XML files contain detailed line and branch coverage information. Key metrics:
- **Line Rate**: Percentage of executable lines covered by tests

These metrics represent the quality of the **mainline branch only**.