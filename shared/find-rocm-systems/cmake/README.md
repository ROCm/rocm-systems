# ROCm Systems CMAKE Config Generation

This document provides an index of all files created for the unified CMake configure for rocm-systems.

## Core Configure Files

### CMakeLists.txt
**Purpose:** Main CMake configuration file  
**Location:** `../CMakeLists.txt`  
**Description:** Root build configuration that orchestrates CMAKE configuration all ROCm systems projects

## Scripts

### CMAKE_Config_gen_rocm-systems.sh
**Purpose:** Linux build automation script  
**Location:** `./CMAKE_Config_gen_rocm-systems.sh`  
**Platform:** Linux/Unix  
**Language:** Bash

**Features:**
- Command-line argument parsing
- Clean build support
- CMAKE configuration File Generation
- Comprehensive help

**Usage:**
```bash
./CMAKE_Config_gen_rocm-systems.sh [options]
./CMAKE_Config_gen_rocm-systems.sh --help
```

**Common options:**
- `--build-dir DIR` - Set build directory
- `--clean` - Clean before building

**Use when:** Building on Linux and you want automation


