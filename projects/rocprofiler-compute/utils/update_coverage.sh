#!/bin/bash
# generate_coverage.sh
# Generate coverage for mainline promotion - stores file only

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
COVERAGE_FILE="$PROJECT_ROOT/coverage/mainline-coverage-latest.xml"

echo "=== Generating Coverage for Mainline Promotion ==="

echo "Setting up clean build..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring with coverage enabled..."
cmake -DENABLE_COVERAGE=ON -DENABLE_TESTS=ON -DPYTEST_NUMPROCS=8 ..

echo "Building..."
make -j$(nproc)

echo "Running ALL tests to generate coverage (this may take a while)..."
ctest --verbose --output-on-failure --parallel 1 || {
    echo "WARNING: Some tests failed, but continuing with coverage generation"
}

echo "Generating coverage report..."
ctest -R generate_coverage_report --output-on-failure --parallel 2

if [ ! -f "coverage.xml" ]; then
    echo "ERROR: coverage.xml not generated"
    exit 1
fi

COVERAGE_INFO=$(python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('coverage.xml')
root = tree.getroot()
line_rate = float(root.get('line-rate', 0)) * 100
print(f'{line_rate:.1f}%')
")

mkdir -p "$PROJECT_ROOT/coverage"
cp coverage.xml "$COVERAGE_FILE"

echo ""
echo "=== Coverage Generated Successfully ==="
echo "Line Coverage: $COVERAGE_INFO"
echo "File: $COVERAGE_FILE"
echo ""
echo "Next steps:"
echo "1. git add $COVERAGE_FILE"
echo "2. git commit -m 'Update coverage for promotion: $COVERAGE_INFO line coverage'"
echo "3. Proceed with staging -> mainline promotion"
echo "4. CDash upload will happen automatically on push"
echo ""
echo "Alternative: To upload to CDash now, run:"
echo "   ctest -D Continuous"