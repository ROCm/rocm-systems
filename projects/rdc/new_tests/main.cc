/*
 * RDC Diagnostics API Test Suite
 *
 * Main entry point for RDC embedded mode API tests.
 * Tests focus on telemetry field collection without requiring rdcd daemon.
 */

#include <gtest/gtest.h>

#include <iostream>

int main(int argc, char** argv) {
  std::cout << "========================================" << std::endl;
  std::cout << "RDC Diagnostics API Test Suite" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Testing in embedded mode (no rdcd daemon required)" << std::endl;
  std::cout << std::endl;

  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Run all tests
  int result = RUN_ALL_TESTS();

  std::cout << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Test Suite Complete" << std::endl;
  std::cout << "========================================" << std::endl;

  return result;
}
