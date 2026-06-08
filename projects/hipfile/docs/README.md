# hipFile Documentation

## Building

These instructions assume you are building on Linux, since hipFile is only supported there.

API documentation can be generated from the header files using CMake. If you enable `AIS_BUILD_DOCS` and have Doxygen installed, CMake will generate HTML, XML, and LaTeX documentation. See `INSTALL.md` in the project root for details.

Sphinx documentation that uses the .rst files from the `docs` directory is built in-place manually. This process will also run Doxygen to build the API documentation from the header files.

### Prerequisites

- Python 3.12+
- [Doxygen](https://www.doxygen.nl/)


### Install Python dependencies

```bash
pip3 install -r ./sphinx/requirements.in
```

### Build HTML documentation

```bash
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Build PDF documentation

PDF output requires a LaTeX installation (e.g. `texlive-full` on Linux).

```bash
sphinx-build -M latexpdf . _build
```

The PDF is written to `_build/latex/hipfile.pdf`.
