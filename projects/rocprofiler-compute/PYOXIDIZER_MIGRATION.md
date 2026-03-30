# Migration from Nuitka to PyOxidizer

## Summary

This document summarizes the migration of the rocprof-compute standalone binary build system from Nuitka to PyOxidizer.

## Changes Made

### 1. New PyOxidizer Configuration (`pyoxidizer.bzl`)

Created a new PyOxidizer configuration file that:
- Defines the build process for creating a standalone binary
- Specifies the entry point as the `rocprof-compute` module
- Includes all dependencies from `requirements.txt`
- Bundles all project packages (rocprof_compute_analyze, rocprof_compute_profile, rocprof_compute_tui, rocprof_compute_soc, utils)
- Embeds VERSION and VERSION.sha files as resources
- Uses in-memory loading for Python modules with filesystem fallback for large resources

### 2. Updated CMakeLists.txt (Line 593)

**Before (Nuitka):**
```cmake
add_custom_target(
    standalonebinary
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src
    COMMAND ${Python3_EXECUTABLE} -m pip list | grep -i nuitka > /dev/null 2>&1
    COMMAND ${Python3_EXECUTABLE} -m pip list | grep -i patchelf > /dev/null 2>&1
    COMMAND git -C ${PROJECT_SOURCE_DIR} rev-parse HEAD > VERSION.sha
    COMMAND ${Python3_EXECUTABLE} -m nuitka --mode=onefile --no-deployment-flag=self-execution ...
    COMMAND patchelf --remove-rpath rocprof-compute.bin
    COMMAND mv rocprof-compute.bin ${CMAKE_BINARY_DIR}
)
```

**After (PyOxidizer):**
```cmake
add_custom_target(
    standalonebinary
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND which pyoxidizer > /dev/null 2>&1 || (echo "ERROR: PyOxidizer not found..." && exit 1)
    COMMAND git -C ${PROJECT_SOURCE_DIR} rev-parse HEAD > ${PROJECT_SOURCE_DIR}/VERSION.sha || echo "unknown" > ${PROJECT_SOURCE_DIR}/VERSION.sha
    COMMAND pyoxidizer build --release
    COMMAND bash -c "find build -type f -name 'rocprof-compute' -executable -exec cp {} ${CMAKE_BINARY_DIR}/rocprof-compute.bin \\;"
    COMMENT "Building standalone binary with PyOxidizer..."
)
```

### 3. Documentation

- `PYOXIDIZER_MIGRATION.md`: This file, documenting the migration (standalone build: see `pyoxidizer.bzl` and `CMakeLists.txt`).

## Key Differences

### Dependencies

| Aspect | Nuitka | PyOxidizer |
|--------|--------|------------|
| Installation | `pip install nuitka patchelf` | `pip install pyoxidizer` |
| Additional Tools | Requires patchelf | No additional tools needed |
| Configuration | Command-line flags | Declarative config file (pyoxidizer.bzl) |

### Build Process

| Aspect | Nuitka | PyOxidizer |
|--------|--------|------------|
| Working Directory | `src/` | Project root |
| Build Command | `python -m nuitka ...` | `pyoxidizer build --release` |
| Post-processing | Manual patchelf for RPATH | Automatic RPATH handling |
| Output Location | Direct to build dir | Find and copy from build tree |

### Advantages of PyOxidizer

1. **No patchelf dependency**: PyOxidizer handles RPATH automatically
2. **Cleaner configuration**: Declarative Starlark syntax vs. long command-line flags
3. **Better resource handling**: More flexible embedding options
4. **Active development**: PyOxidizer is actively maintained
5. **Performance**: Generally faster startup and runtime
6. **Simpler build process**: Single command instead of multiple steps

## Compatibility

The migration maintains full compatibility:
- Same output binary name: `rocprof-compute.bin`
- Same dependencies included
- Same functionality
- Same installation structure

## Testing

To verify the migration works correctly:

```bash
# Build the binary
mkdir build && cd build
cmake ..
cmake --build . --target standalonebinary

# Test the binary
./rocprof-compute.bin --help
./rocprof-compute.bin profile --help
./rocprof-compute.bin analyze --help
```

## Migration Checklist

- [x] Create PyOxidizer configuration file (`pyoxidizer.bzl`)
- [x] Update CMakeLists.txt to use PyOxidizer
- [x] Remove Nuitka-specific commands (patchelf)
- [x] Update working directory to project root
- [x] Add error handling for missing PyOxidizer
- [x] Create build documentation
- [x] Document migration process
- [ ] Test binary build on target systems
- [ ] Update CI/CD pipelines (if applicable)
- [ ] Update developer documentation

## Rollback Plan

If issues arise, the Nuitka configuration can be restored from git history:

```bash
git show HEAD~1:CMakeLists.txt > CMakeLists.txt.nuitka
# Review and restore if needed
```

## Future Improvements

Potential enhancements for the PyOxidizer build:

1. **Optimize binary size**: Fine-tune which resources are embedded vs. filesystem
2. **Cross-compilation**: Add support for building on different platforms
3. **Custom Python distribution**: Use a minimal Python distribution for smaller binaries
4. **Resource compression**: Enable compression for embedded resources
5. **Multi-platform builds**: Add configurations for Windows and macOS

## References

- [PyOxidizer Documentation](https://pyoxidizer.readthedocs.io/)
- [PyOxidizer GitHub](https://github.com/indygreg/PyOxidizer)
- [Nuitka Documentation](https://nuitka.net/doc/user-manual.html) (for reference)
## Contact

For questions or issues related to this migration, please file an issue in the rocprof-compute repository.
