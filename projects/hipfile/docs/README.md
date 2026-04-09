# hipFile Documentation

hipFile is documented using Doxygen markup in the public header
files and Sphinx .rst files.

## Building the Documentation

hipFile can generate two documentation products:

1. API documentation
2. Sphinx HTML documentation (includes #1)

API documentation is generated from Doxygen and limited to markup
in the header files. This will generate HTML, XML, and LaTeX output.
The LaTeX can be used to create a pdf document.

The Sphinx documentation incorporates the Doxygen API markup and
adds reStructured text file content to generate HTML documentation
for the web.

The documentation CMake target currently builds both. There is
no way to select "just Doxygen".

### Requirements

* CMake >= 3.21
* Doxygen (we use 1.9.8, other versions are untested)
* Python 3.12 (earlier versions might work but are untested)
* The following Python packages:
    * breathe
    * rocm-docs-core
    * sphinx
    * sphinx-rtd-theme

If you want to build a pdf, you will need LaTeX and pdflatex.

### Generation

Building the documentation is done via CMake by turning on the docs
option (which is off by default).

    `cmake -DAIS_BUILD_DOCS=ON <options> <path/to/source>`

You can then build the `doc` target.

    `cmake --build . --target doc`

The documentation will be in the `docs` subdirectory of the build
directory:

```
    ├── doxygen
    │   ├── html
    │   ├── latex
    │   └── xml
    └── sphinx
        └── html
```

To generate a pdf for the API docs, navigate to `doxygen/latex` and run `make pdf` to
generate a pdf named `refman.pdf`.

To generate a pdf for the Sphinx docs, navigate to `sphinx/html` and run `make` in
the directory. This will generate a pdf named `rocshmem.pdf` (we are stealing their
configs until `rocm-docs-core` is updated).

## Adding to the Documentation

### API Documentation (Doxygen)

* Make your markup look like the rest of the file
* Doxygen macros can be found in `docs/doxygen/Doxyfile.in`
* All public API types, functions, etc. MUST have Doxygen markup

### Other Documentation (Sphinx)

All other documentation goes in Sphinx .rst documents. API-specific documentation
belongs in `docs/API` and other documentation goes in `docs`. Be sure to update
`sphinx/_toc.yml.in` if you add any new files.
