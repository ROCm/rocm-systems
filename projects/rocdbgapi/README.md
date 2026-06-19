# ROCdbgapi-docs

> [!NOTE]
> The published documentation is available at [ROCdbgapi documentation](https://rocm.docs.amd.com/projects/ROCdbgapi/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents.

## Important files from the submodule

The header input file needs to be processed

`ROCdbgapi/include/amd-dbgapi.h.in`

```bash
# update submodule
cd ROCdbgapi
git checkout <branch>
git pull
cd ..
# copy important files from submodule
cp ROCdbgapi/include/amd-dbgapi.h.in ./include/
# process important files from submodule
cmake .
make
```

## How to build documentation locally

Run the following steps to build the base documentation site:

```bash
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```
