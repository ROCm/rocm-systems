#!/bin/bash

# Script to add prerequisite checks to all conftest.py files
# This is a helper script for systematic test improvement

echo "Updating conftest.py files with prerequisite checks..."

for conftest in $(find projects/rocprofiler-sdk/tests -name "conftest.py"); do
    # Skip if already has pytest_configure
    if grep -q "def pytest_configure" "$conftest"; then
        echo "  [SKIP] $conftest - already has pytest_configure"
        continue
    fi

    # Skip if doesn't import pytest
    if ! grep -q "import pytest" "$conftest"; then
        echo "  [SKIP] $conftest - doesn't import pytest"
        continue
    fi

    echo "  [UPDATE] $conftest"

    # Create backup
    cp "$conftest" "${conftest}.bak"

    # Add the import and pytest_configure after the imports section
    awk '
    BEGIN { imported = 0; added = 0 }
    /^import pytest/ { imported = 1 }
    /^from pytest_utils import/ { next }  # Skip if already there
    imported && !added && /^$/ || /^def / {
        if (!added) {
            print "from pytest_utils import check_test_prerequisites"
            print ""
            print ""
            print "def pytest_configure(config):"
            print "    \"\"\"Run prerequisite checks before any tests\"\"\""
            print "    error = check_test_prerequisites()"
            print "    if error:"
            print "        pytest.exit(error, returncode=1)"
            print ""
            print ""
            added = 1
        }
    }
    { print }
    ' "${conftest}.bak" > "$conftest"
done

echo "Done updating conftest.py files"
echo "Backups created with .bak extension"
