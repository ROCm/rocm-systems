#!/usr/bin/env python3

"""
Repository Config Model
------------------------

This module defines Pydantic data models for validating and parsing the repos-config.json file.

Structure of the expected JSON:

{
    "repositories": [
        {
            "name": "rocblas",
            "url": "ROCm/rocBLAS",
            "branch": "develop",
            "category": "projects"
        },
        ...
    ]
}
"""

from typing import List
from pydantic import BaseModel

class RepoEntry(BaseModel):
    """
    Represents a single repository entry in the repos-config.json file.

    Fields:
        name     : Name of the project matching packaging file names. Lower-cased and no underscores. (e.g., "rocblas")
        url      : Individual GitHub org plus repo names in matching case and punctuation. (e.g., "ROCm/rocBLAS")
        branch   : The base branch of the sub-repo to target (e.g., "develop").
        category : Directory category in the super-repo (e.g., "projects" or "shared").
    """
    name: str
    url: str
    branch: str
    category: str
    auto_subtree_pull: bool
    auto_subtree_push: bool

class RepoConfig(BaseModel):
    """
    Represents the full config file structure.

    Fields:
        repositories : List of RepoEntry items.
    """
    repositories: List[RepoEntry]
