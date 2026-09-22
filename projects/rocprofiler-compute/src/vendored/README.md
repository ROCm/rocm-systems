# Vendored Dependencies

This directory contains external Python packages that are vendored (included directly) in the rocprofiler-compute source tree.

## Current Vendored Packages

| Package | Version | License | Source | Vendored |
|---------|---------|---------|--------|----------|
| PyYAML  | 6.0.3   | MIT     | https://github.com/yaml/pyyaml | 2024-03-14 |

## Usage

```python
# Recommended: Use convenience wrapper
from vendored import yaml

# Alternative: Direct import
from vendored.pyyaml.lib import yaml
```

## Why Vendor?

To eliminate external dependencies in profile mode, improving:
- **Portability**: Works without pip/internet access on HPC systems
- **Reliability**: No version conflicts with user environments
- **Installation**: Simpler deployment, no separate pip install step

---

## Vendoring Workflow

### What We Vendor

**Criteria for vendoring:**
- Pure Python packages only (no C extensions)
- Critical dependencies used in profile code path
- Stable, mature packages with permissive licenses (MIT, BSD, Apache 2.0)

**Do NOT vendor:**
- Test-only dependencies (use `requirements-test.txt` instead)
- Analyze-mode only dependencies (use `requirements.txt` instead)
- Packages with C extensions (portability issues)

### Adding a New Vendored Package

Follow these steps to vendor a new package (example: hypothetical future package):

#### 1. Add as Git Submodule

```bash
cd /path/to/rocprofiler-compute

# Add the package as a submodule under src/vendored/
git submodule add https://github.com/org/package.git src/vendored/package

# Pin to a specific version
cd src/vendored/package
git checkout v1.2.3  # Use appropriate version tag
cd ../../..

git add .gitmodules src/vendored/package
git commit -m "Vendor package v1.2.3"
```

#### 2. Update `src/vendored/__init__.py`

Add a convenience import for the new package:

```python
# Package v1.2.3
# Source: https://github.com/org/package (License)
# Vendored: YYYY-MM-DD
from .package.lib import module

__all__ = ['yaml', 'module']  # Add to __all__
```

#### 3. Add CMake Install Block

In `CMakeLists.txt`, add a new install block in the "Vendored Dependencies Installation" section:

```cmake
# Package v1.2.3 (pure Python only, no C extensions)
install(
    DIRECTORY src/vendored/package/lib/module
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/vendored/package/lib
    COMPONENT main
    FILES_MATCHING PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.so" EXCLUDE      # Exclude any C extensions
    PATTERN "*.pyd" EXCLUDE     # Exclude Windows extensions
)
```

**Important**: Each vendored package has unique structure. Adjust the `DIRECTORY` path and patterns based on the package's layout. Only install pure Python files (`.py`).

#### 4. Add Submodule Auto-Init to CMake

In the "Git Submodule Auto-Initialization" section of `CMakeLists.txt`, add a check for the new submodule:

```cmake
# Package submodule
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/src/vendored/package/.git")
    message(STATUS "  Initializing vendored/package submodule...")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init src/vendored/package
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_SUBMOD_RESULT
    )
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update failed for src/vendored/package")
    endif()
else()
    message(STATUS "  vendored/package: OK")
endif()
```

#### 5. Update Documentation

- Update the table above with package details (name, version, license, source URL, date)

### Updating a Vendored Package

```bash
# Example: Update PyYAML from 6.0.3 to 6.0.4
cd src/vendored/pyyaml
git fetch origin
git checkout 6.0.4  # New version tag
cd ../../..

# Update version in src/vendored/README.md table
# Version update might require changes in how CMake installs that vendored package

git add src/vendored/pyyaml src/vendored/README.md
git commit -m "Update vendored PyYAML to 6.0.4"
```

### Using Vendored Packages in Code

```python
# Recommended: Use convenience wrapper
from vendored import yaml

# Alternative: Direct import
from vendored.pyyaml.lib import yaml
```

The convenience wrapper (`from vendored import yaml`) is preferred as it provides a stable import interface even if the underlying package structure changes.
