# Building the rocSHMEM documentation

## Prerequisites

- Python 3.10+
- [Doxygen](https://www.doxygen.nl/)

## Linux

Install system dependencies:

```bash
sudo apt-get install -y doxygen
```

Install Python dependencies:

```bash
pip3 install -r ./sphinx/requirements.txt
```

Build HTML documentation:

```bash
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Open the generated docs:

```bash
xdg-open _build/html/index.html
```

### Building to a custom output directory

```bash
python3 -m sphinx -T -E -b html -d <output-dir>/doctrees -D language=en . <output-dir>/html
```

## macOS

Install system dependencies:

```bash
brew install doxygen sphinx-doc
```

Install Python dependencies and build:

```bash
pip3 install -r ./sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
open _build/html/index.html
```

## PDF output

PDF output requires a LaTeX installation (e.g. `texlive-full` on Linux, MacTeX on macOS).

```bash
pip3 install -r ./sphinx/requirements.txt
sphinx-build -M latexpdf . _build
```

The PDF is written to `_build/latex/rocshmem.pdf`.
