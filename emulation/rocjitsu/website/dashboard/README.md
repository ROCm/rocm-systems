# Rocjitsu Simulation Performance Dashboard

React + Vite dashboard for browsing benchmark history and comparing runs, with MUI and ECharts.

[Live dashboard](https://rocm.github.io/rocm-systems/rocjitsu-dashboard/)

## Quick start

Use a Node.js version matching [package.json](package.json). From the repository root:

```bash
cd emulation/rocjitsu/website/dashboard
npm ci
npm run build -- --mode fixtures
npm run preview -- --mode fixtures
```

Open `http://localhost:4174`. This preview uses fictional test data, not production measurements.

## Documentation

- [Build and test](docs/build-and-test.md): setup, commands, and test execution.
- [Website and publication](docs/architecture-and-publication.md): source map, deployment, data fetching, publishing workflows, and immutability.
- [Schema 1](docs/schema-1.md): legacy JSON layout.
- [Schema 2](docs/schema-2.md): current JSON layout and field constraints.
- [Schema migration](docs/schema-migration.md): version detection, exclusions, and per-target ST/MT mapping.
- [Test fixtures](tests/fixtures/README.md): fictional datasets and test cases.
- [Publishing setup](../handbook/README.md#verification-and-publishing): GitHub App credentials and Pages configuration.
