#!/bin/bash

# Demo script for ROCm Systems Profiler User Experience Improvements
# This script demonstrates the new features added to improve first-time user experience

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ROCm Systems Profiler - User Experience Improvements Demo      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to show a demo section
show_demo() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Function to show a command
show_command() {
    echo -e "${BLUE}$ $1${NC}"
    echo ""
}

# Check if binaries are built
check_binary() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${YELLOW}Warning: $1 not found in PATH${NC}"
        echo "Please build the project first:"
        echo "  cmake -B build"
        echo "  cmake --build build"
        echo ""
        return 1
    fi
    return 0
}

# Demo 1: Quick Reference Card
show_demo "Demo 1: Quick Reference Card (--cheatsheet)"
echo "The cheatsheet provides a one-page reference for common commands:"
echo ""
show_command "rocprof-sys-sample --cheatsheet"
echo "Press Enter to see the cheatsheet..."
read -r
if check_binary rocprof-sys-sample; then
    rocprof-sys-sample --cheatsheet
fi

# Demo 2: Standardized Help
show_demo "Demo 2: Standardized Help Text"
echo "All binaries now have consistent, tiered examples:"
echo ""
show_command "rocprof-sys-sample --help | head -40"
echo "Press Enter to see the help text..."
read -r
if check_binary rocprof-sys-sample; then
    rocprof-sys-sample --help | head -40
fi

# Demo 3: Preset Validation
show_demo "Demo 3: Preset Mode Validation"
echo "Trying to use multiple conflicting presets:"
echo ""
show_command "rocprof-sys-sample --quick --simple -- ls"
echo "Press Enter to see the validation error..."
read -r
if check_binary rocprof-sys-sample; then
    rocprof-sys-sample --quick --simple -- ls 2>&1 || true
fi

# Demo 4: Pre-execution Information
show_demo "Demo 4: Pre-execution Information"
echo "Using a preset shows where results will be saved:"
echo ""
show_command "rocprof-sys-sample --quick -- ls"
echo "Press Enter to see the pre-execution info..."
read -r
if check_binary rocprof-sys-sample; then
    echo "Note: This will actually profile 'ls' command"
    rocprof-sys-sample --quick -- ls 2>&1 | head -20
fi

# Demo 5: Interactive Wizard (simulated)
show_demo "Demo 5: Interactive Wizard"
echo "The wizard helps first-time users choose the right options:"
echo ""
show_command "rocprof-sys-sample --wizard"
echo ""
echo "The wizard would ask:"
echo "  1. What type of application? (HIP/GPU, HPC, CPU, Python)"
echo "  2. Quick or detailed profiling?"
echo ""
echo "Then provides a personalized command recommendation."
echo ""
echo "Press Enter to run the actual wizard (you'll need to interact)..."
read -r
if check_binary rocprof-sys-sample; then
    echo "Starting wizard (press Ctrl+C to skip):"
    rocprof-sys-sample --wizard || true
fi

# Demo 6: All Binaries Have Same Features
show_demo "Demo 6: Consistent Across All Tools"
echo "All three binaries have the same user experience features:"
echo ""
echo "rocprof-sys-sample   - Sampling profiler"
echo "rocprof-sys-run      - Run instrumented binaries"
echo "rocprof-sys-instrument - Binary instrumentation"
echo ""
echo "All support:"
echo "  --cheatsheet  : Quick reference"
echo "  --wizard      : Interactive setup"
echo "  --quick       : Quick profiling preset"
echo "  --trace-hpc   : HPC workload preset"
echo "  --trace-ai    : AI/ML workload preset"
echo ""

# Summary
show_demo "Summary of Improvements"
echo "✅ 1. Pre-execution information messages"
echo "   → Shows where results will be saved"
echo "   → Warns about potential issues"
echo ""
echo "✅ 2. Preset mode validation"
echo "   → Prevents conflicting options"
echo "   → Clear error messages"
echo ""
echo "✅ 3. Error guidance with solutions"
echo "   → Context-specific help"
echo "   → Actionable troubleshooting steps"
echo ""
echo "✅ 4. Standardized help text"
echo "   → Beginner/Intermediate/Advanced tiers"
echo "   → Consistent across all tools"
echo ""
echo "✅ 5. Quick reference card (--cheatsheet)"
echo "   → One-page reference"
echo "   → Common commands and workflow"
echo ""
echo "✅ 6. Interactive wizard (--wizard)"
echo "   → Personalized recommendations"
echo "   → Guided setup for first-time users"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "For more information, see:"
echo "  • USER_EXPERIENCE_IMPROVEMENTS.md"
echo "  • examples/README.md"
echo "  • docs/tutorials/quickstart.rst"
echo ""
