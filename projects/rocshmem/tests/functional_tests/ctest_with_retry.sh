#!/bin/bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# CTest wrapper with automatic retry for failed tests
# This mimics the driver.sh retry behavior

# Configuration
RETRY_THRESHOLD=${RETRY_THRESHOLD:-5}  # Max number of tests to retry
CTEST_ARGS="$@"  # Pass through all arguments to ctest

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running CTest with automatic retry${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Run tests first time
echo "Running tests..."
ctest $CTEST_ARGS
FIRST_RUN_STATUS=$?

# Check if there were failures
if [ $FIRST_RUN_STATUS -ne 0 ]; then
    # Count failed tests
    if [ -f Testing/Temporary/LastTestsFailed.log ]; then
        FAILED_COUNT=$(wc -l < Testing/Temporary/LastTestsFailed.log)

        echo ""
        echo -e "${YELLOW}========================================${NC}"
        echo -e "${YELLOW}${FAILED_COUNT} test(s) failed${NC}"
        echo -e "${YELLOW}========================================${NC}"

        # Check if below retry threshold
        if [ $FAILED_COUNT -le $RETRY_THRESHOLD ]; then
            echo ""
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}Retrying ${FAILED_COUNT} failed test(s)...${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo ""

            # Rerun failed tests
            ctest --rerun-failed $CTEST_ARGS
            RETRY_STATUS=$?

            # Summary
            echo ""
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}Retry Summary${NC}"
            echo -e "${BLUE}========================================${NC}"

            if [ $RETRY_STATUS -eq 0 ]; then
                echo -e "${GREEN}All tests passed on retry!${NC}"
                exit 0
            else
                if [ -f Testing/Temporary/LastTestsFailed.log ]; then
                    STILL_FAILED=$(wc -l < Testing/Temporary/LastTestsFailed.log)
                    PASSED_ON_RETRY=$((FAILED_COUNT - STILL_FAILED))

                    if [ $PASSED_ON_RETRY -gt 0 ]; then
                        echo -e "${GREEN}${PASSED_ON_RETRY} test(s) passed on retry${NC}"
                    fi

                    if [ $STILL_FAILED -gt 0 ]; then
                        echo -e "${RED}${STILL_FAILED} test(s) still failed after retry${NC}"
                        echo ""
                        echo "Failed tests:"
                        cat Testing/Temporary/LastTestsFailed.log
                    fi
                fi
                exit 1
            fi
        else
            echo ""
            echo -e "${YELLOW}========================================${NC}"
            echo -e "${YELLOW}Too many failures (${FAILED_COUNT} > ${RETRY_THRESHOLD})${NC}"
            echo -e "${YELLOW}Skipping retry - this may indicate a systemic issue${NC}"
            echo -e "${YELLOW}========================================${NC}"
            echo ""
            echo "To retry anyway, run:"
            echo "  ctest --rerun-failed $CTEST_ARGS"
            echo ""
            echo "Or increase threshold:"
            echo "  RETRY_THRESHOLD=10 $0 $CTEST_ARGS"
            exit 1
        fi
    else
        # No LastTestsFailed.log but non-zero exit
        echo -e "${RED}Tests failed but no failure log found${NC}"
        exit $FIRST_RUN_STATUS
    fi
else
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
fi
