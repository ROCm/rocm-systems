# Building the rocSHMEM documentation

## macOS

To build html documentation locally:

```
brew install doxygen sphinx-doc
pip3.10 install -r ./sphinx/requirements.txt
python3.10 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
open _build/html/index.html
```

To build pdf documentation we require a LaTeX installation on your machine.
Once LaTeX is installed, you may run the following:

```
pip3.10 install -r ./sphinx/requirements.txt
sphinx-build -M latexpdf . _build
open _build/latex/rocshmem.pdf
```
