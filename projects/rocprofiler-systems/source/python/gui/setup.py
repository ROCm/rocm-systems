#!/usr/bin/env python3

import os

from setuptools import setup


def get_project_version():
    # open "VERSION"
    _cwd = os.path.dirname(__file__)
    with open(os.path.join(_cwd, "source", "VERSION"), "r") as f:
        data = f.read().replace("\n", "")
    # make sure is string
    if isinstance(data, list) or isinstance(data, tuple):
        return data[0]
    else:
        return data


def get_long_description():
    long_descript = ""
    try:
        long_descript = open("README.md").read()
    except Exception:
        long_descript = ""
    return long_descript


def parse_requirements(fname="requirements.txt"):
    _req = []
    requirements = []
    # read in the initial set of requirements
    with open(fname, "r") as fp:
        _req = list(filter(bool, (line.strip() for line in fp)))
    # look for entries which read other files
    for itr in _req:
        if itr.startswith("-r "):
            # read another file
            for fitr in itr.split(" "):
                if os.path.exists(fitr):
                    requirements.extend(parse_requirements(fitr))
        else:
            # append package
            requirements.append(itr)
    # return the requirements
    return requirements


setup(
    name="rocprof-sys-causal-viewer",
    version=get_project_version(),
    description="GUI for viewing causal profilers",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="AMD Research",
    packages=["rocprof_sys_causal_viewer"],
    package_dir={"rocprof_sys_causal_viewer": "source"},
    package_data={
        "rocprof_sys_causal_viewer": [
            "source/assets/*",
            "source/workloads/*",
            "source/VERSION",
        ]
    },
    install_requires=parse_requirements(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "rocprof-sys-causal-plot=rocprof_sys_causal_viewer.__main__:main"
        ]
    },
)
