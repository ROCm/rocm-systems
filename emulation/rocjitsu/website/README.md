# Rocjitsu Performance Dashboard

React + Vite source for the Rocjitsu benchmark dashboard, with MUI components and
ECharts visualizations. This directory contains application source, build configuration,
and tests. The intended release flow copies the built application to a separate
GitHub Pages deployment branch, where real benchmark data will be published
independently. Pages deployment is not enabled by this source package.

## Quick start: preview with dummy data

Use npm and a Node.js version matching the `engines` field in [package.json](package.json).
From the `rocm-systems` repository root:

```bash
cd emulation/rocjitsu/website
npm ci
npm run build -- --mode fixtures
npm run preview -- --mode fixtures
```

Open http://localhost:4173 to visualize the website with dummy benchmark data.
Press **Ctrl+C** to stop the preview. The fixture build uses `.test-dist/`.

For deployment, `npm run build` produces application files in `dist/` without dummy
data. The browser loads real benchmark JSON from the deployed site's `data/` directory.

See the [build and test guide](docs/build-and-test.md) for local development with
dummy data, browser setup, verification commands, and the deployment branch handoff.

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
| `docs/` | Build, testing, deployment, and data-contract documentation |

## Further documentation

- [Website data contract](docs/website-data-contract.md): JSON schema, comparison
  rules, and data publication requirements.
- [Test fixtures](tests/fixtures/README.md): focused test cases and generated history.
