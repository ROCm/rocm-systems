# Rocjitsu Simulation Performance Dashboard

React + Vite source for the Rocjitsu simulation-performance dashboard, with MUI components and
ECharts visualizations. This directory contains application source, build configuration,
and tests. Production builds fetch benchmark JSON from `rocjitsu-dashboard/data/` on
the `gh-pages-rocjitsu branch` in the `ROCm/rocm-systems` repository.

**Live website:** [RocJitsu Performance Dashboard](https://rocm.github.io/rocm-systems/rocjitsu-dashboard/)

New runs use **schema 2**, with explicitly declared ST/MT
`configurations[].threadingMode` values. The website explicitly migrates legacy
schema-1 Vanilla target groups to MT and excludes non-Vanilla plugin files.
Other incompatible measurements fail the load instead of being silently skipped. The local fixtures are fictional test input, not
performance measurements; passing fixture tests does not verify a live publication. See the
[schema-2 data contract](docs/website-data-contract.md) for fields, normalization,
branch-reference rules, local processing and future publisher acceptance. This
package includes the prepare/publish staging scripts under `scripts/dashboard-data/`;
benchmark execution and deployment remain separate responsibilities.

## Quick start: preview with dummy data

Use npm and a Node.js version matching the `engines` field in [package.json](package.json).
From the `rocm-systems` repository root:

```bash
cd emulation/rocjitsu/website/dashboard
npm ci
npm run build -- --mode fixtures
npm run preview -- --mode fixtures
```

Open http://localhost:4174 to visualize the website with dummy benchmark data.
Press **Ctrl+C** to stop the preview. The fixture build uses `.test-dist/` and the
fixture preview always uses port 4174; the production preview uses port 4173.

The production build writes application files to `dist/` without dummy data and loads
JSON from the [`gh-pages-rocjitsu` branch](https://github.com/ROCm/rocm-systems/tree/gh-pages-rocjitsu/rocjitsu-dashboard/data).
The data directory must contain `index.json` with `generatedAt`, `runFiles`,
and the referenced run/catalog resources. New runs declare numeric
`schemaVersion: 2` individually and live in `runs/default-branch/` (develop) or
flat `runs/side-branches/`. Indexes and catalogs carry no schema version. An empty
publication with no indexed runs does not require `test-catalogs/` or `runs/`
directories.

Site settings and the target run schema version come from source-controlled
`src/config/metadata.json`, imported by `src/config/siteConfig.js` and bundled in
production, fixture and local-data builds. Changing settings requires rebuilding
the website; publication-side legacy `metadata.json` is ignored, not automatically deleted.
See the [rollout procedure](docs/data-generation-guide.md#5-bundled-site-config-rollout)
before publishing the new website against existing data. No live deployment is
performed by this refactor.

Before any future data publication, validate the complete staged data directory:

```bash
npm run validate:data -- /absolute/path/to/staged/data
```

Do not publish when this command fails. The browser uses the same validator and
fails closed instead of displaying partial history when invalid data bypasses the
publication gate.

See the [build and test guide](docs/build-and-test.md) for local development with
dummy data or a local data directory (`npm run dev:data -- <data-directory>`),
browser setup, verification commands, and the production build.

## Publishing to GitHub Pages

The [rocjitsu-publish-website workflow](../../../../.github/workflows/rocjitsu-publish-website.yml)
compares each site's sources on `develop` with its last published revision and
builds only sites with pending changes. Manual runs and shared workflow changes
rebuild both. Manual publishing also requires `develop`.
The dashboard is published to `gh-pages/rocjitsu-dashboard/`, and the handbook
to `gh-pages/rocjitsu/`; other Pages files are preserved. Benchmark data
continues to come from `gh-pages-rocjitsu`.

See the [handbook publishing setup](../handbook/README.md#verification-and-publishing)
for GitHub App secrets, permissions, and Pages configuration.

## Source layout

| Path | Purpose |
| --- | --- |
| `index.html`, `src/main.jsx` | Browser entry points |
| `src/components/`, `src/hooks/` | Views, shared components, interaction state |
| `src/data/` | Data loading, validation, selectors, comparison logic |
| `src/config/metadata.json`, `src/config/siteConfig.js` | Bundled target run schema, repository URL, Beta flag and canonical branch |
| `src/theme/`, `src/utils/`, `src/index.css` | Theme, chart/display helpers, styles |
| `package.json`, `package-lock.json` | Commands and reproducible dependency installation |
| `*.config.js` | Vite, ESLint, Vitest, and Playwright configuration |
| `tests/unit/`, `tests/e2e/`, `tests/fixtures/` | Tests, helpers, and dummy fixtures |
| `docs/` | Build, testing, hosting, and data-contract documentation |

## Further documentation

- [Documentation index](docs/README.md): maintained dashboard documentation.
- [Data-generation guide](docs/data-generation-guide.md): practical starting point
  for workflow authors generating and staging website input.
- [Website data contract](docs/website-data-contract.md): JSON schema, comparison
  rules, and data publication requirements.
- [Test fixtures](tests/fixtures/README.md): focused test cases and generated history.
