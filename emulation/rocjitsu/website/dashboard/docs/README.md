# Dashboard documentation

- [Data-generation guide](data-generation-guide.md): start here when designing a
  benchmark workflow that supplies the website. Covers staging, field mapping,
  validation, local preview and the publication boundary.
- [Data contract](website-data-contract.md): authoritative JSON fields,
  complete fictional examples, identity rules, consumer calculations and publisher
  acceptance requirements.
- [Build and test](build-and-test.md): dependency setup, development, verification
  commands and production builds.

Workflow authors must supply measured results that satisfy the data contract.
Fixtures are test input, not a publishable dataset. Producer and publisher
implementation is outside this frontend package.
