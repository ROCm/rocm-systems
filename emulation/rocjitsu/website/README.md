# Rocjitsu Simulation Performance Dashboard

React + Vite source for the Rocjitsu simulation-performance dashboard, with MUI components and
ECharts visualizations. This directory contains application source, build configuration,
and tests. Production builds fetch benchmark JSON from `rocjitsu-dashboard/data/` on
the `gh-pages-rocjitsu branch` in the `ROCm/rocm-systems` repository.

**Live website:** [RocJitsu Performance Dashboard](https://rocm.github.io/rocm-systems/rocjitsu-dashboard/)

## Quick start: preview with dummy data

Use npm and a Node.js version matching the `engines` field in [package.json](package.json).
From the `rocm-systems` repository root:

```bash
cd emulation/rocjitsu/website
npm ci
npm run build -- --mode fixtures
npm run preview -- --mode fixtures
```

Open http://localhost:4174 to visualize the website with dummy benchmark data.
Press **Ctrl+C** to stop the preview. The fixture build uses `.test-dist/` and the
fixture preview always uses port 4174; the production preview uses port 4173.

The production build writes application files to `dist/` without dummy data and loads
JSON from the [`gh-pages-rocjitsu` branch](https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/gh-pages-rocjitsu/rocjitsu-dashboard/data/).
The data directory must contain `metadata.json`, `index.json`, `test-catalogs/`, and `runs/`.
Before every data publication, validate the complete staged data directory:

```bash
npm run validate:data -- /absolute/path/to/staged/data
```

Do not publish when this command fails. The browser assumes published input is valid
and fails closed instead of displaying partial history when invalid data bypasses the
publication gate.

See the [build and test guide](docs/build-and-test.md) for local development with
dummy data or a local data directory (`npm run dev:data -- <data-directory>`),
browser setup, verification commands, and the production build.

## Source layout

| Path | Purpose |
| --- | --- |
| `index.html`, `src/main.jsx` | Browser entry points |
| `src/components/`, `src/hooks/` | Views, shared components, interaction state |
| `src/data/` | Data loading, validation, selectors, comparison logic |
| `src/theme/`, `src/utils/`, `src/index.css` | Theme, chart/display helpers, styles |
| `package.json`, `package-lock.json` | Commands and reproducible dependency installation |
| `*.config.js` | Vite, ESLint, Vitest, and Playwright configuration |
| `tests/unit/`, `tests/e2e/`, `tests/fixtures/` | Tests, helpers, and dummy fixtures |
| `docs/` | Build, testing, hosting, and data-contract documentation |

## Further documentation

- [Website data contract](docs/website-data-contract.md): JSON schema, comparison
  rules, and data publication requirements.
- [Test fixtures](tests/fixtures/README.md): focused test cases and generated history.
