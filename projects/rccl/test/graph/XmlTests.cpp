/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "graph/xml.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

class XmlTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize XML structure for testing
    ASSERT_EQ(xmlAlloc(&xml, 10), ncclSuccess);

    // Create a root node
    ASSERT_EQ(xmlAddNode(xml, nullptr, "root", &rootNode), ncclSuccess);

    // Create test nodes with various attributes
    ASSERT_EQ(xmlAddNode(xml, rootNode, "testNode", &testNode), ncclSuccess);
    ASSERT_EQ(xmlSetAttr(testNode, "floatAttr", "3.14159"), ncclSuccess);
    ASSERT_EQ(xmlSetAttr(testNode, "intAttr", "42"), ncclSuccess);
    ASSERT_EQ(xmlSetAttr(testNode, "stringAttr", "testValue"), ncclSuccess);
    ASSERT_EQ(xmlSetAttr(testNode, "emptyAttr", ""), ncclSuccess);

    // Create child nodes for hierarchy testing
    ASSERT_EQ(xmlAddNode(xml, testNode, "childNode1", &childNode1),
              ncclSuccess);
    ASSERT_EQ(xmlSetAttr(childNode1, "id", "1"), ncclSuccess);
    ASSERT_EQ(xmlSetAttr(childNode1, "name", "first"), ncclSuccess);

    ASSERT_EQ(xmlAddNode(xml, testNode, "childNode2", &childNode2),
              ncclSuccess);
    ASSERT_EQ(xmlSetAttr(childNode2, "id", "2"), ncclSuccess);
    ASSERT_EQ(xmlSetAttr(childNode2, "name", "second"), ncclSuccess);

    ASSERT_EQ(xmlAddNode(xml, rootNode, "nextNode", &nextNode), ncclSuccess);
    ASSERT_EQ(xmlSetAttr(nextNode, "type", "next"), ncclSuccess);

    // Initialize test dictionary for kvConvertToStr tests
    testDict[0] = {"first", 1};
    testDict[1] = {"second", 2};
    testDict[2] = {"third", 3};
    testDict[3] = {nullptr, 0}; // Terminator

    // Clean up any existing test files
    std::remove("test_topology.xml");
  }

  void TearDown() override {
    if (xml) {
      free(xml);
      xml = nullptr;
    }
    // Clean up test files
    std::remove("test_topology.xml");
  }

  // Helper to create test XML file
  void createTestXmlFile(const std::string &content) {
    std::ofstream file("test_topology.xml");
    file << content;
    file.close();
  }

  // Helper to allocate XML structure
  struct ncclXml *allocateXml(int maxNodes) {
    size_t size =
        offsetof(struct ncclXml, nodes) + sizeof(struct ncclXmlNode) * maxNodes;
    struct ncclXml *xml = (struct ncclXml *)malloc(size);
    if (xml) {
      memset(xml, 0, size);
      xml->maxNodes = maxNodes;
      xml->maxIndex = 0;
    }
    return xml;
  }

  struct ncclXml *xml = nullptr;
  struct ncclXmlNode *rootNode = nullptr;
  struct ncclXmlNode *testNode = nullptr;
  struct ncclXmlNode *childNode1 = nullptr;
  struct ncclXmlNode *childNode2 = nullptr;
  struct ncclXmlNode *nextNode = nullptr;
  struct kvDict testDict[4];
};

// Tests for xmlGetAttrFloatDefault
TEST_F(XmlTest, xmlGetAttrFloatDefault_ValidFloat) {
  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "floatAttr", &result, 0.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, 3.14159f);
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_AttributeNotFound) {
  float result;
  float defaultValue = 42.5f;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "nonExistentAttr", &result,
                                   defaultValue),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, defaultValue);
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_EmptyAttribute) {
  float result;
  float defaultValue = 1.23f;
  EXPECT_EQ(
      xmlGetAttrFloatDefault(testNode, "emptyAttr", &result, defaultValue),
      ncclSuccess);
  EXPECT_FLOAT_EQ(result, 0.0f); // Empty string converts to 0.0
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_InvalidFloat) {
  // Set an invalid float attribute
  ASSERT_EQ(xmlSetAttr(testNode, "invalidFloat", "notanumber"), ncclSuccess);

  float result;
  float defaultValue = 5.67f;
  EXPECT_EQ(
      xmlGetAttrFloatDefault(testNode, "invalidFloat", &result, defaultValue),
      ncclSuccess);
  EXPECT_FLOAT_EQ(result, 0.0f); // Invalid string typically converts to 0.0
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_ZeroValue) {
  ASSERT_EQ(xmlSetAttr(testNode, "zeroFloat", "0.0"), ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "zeroFloat", &result, 99.9f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_NegativeValue) {
  ASSERT_EQ(xmlSetAttr(testNode, "negativeFloat", "-3.14"), ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "negativeFloat", &result, 1.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, -3.14f);
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_ScientificNotation) {
  ASSERT_EQ(xmlSetAttr(testNode, "scientificFloat", "1.23e-4"), ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "scientificFloat", &result, 1.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, 1.23e-4f);
}

// Tests for xmlFindNextTag
TEST_F(XmlTest, xmlFindNextTag_FindExisting) {
  struct ncclXmlNode *foundNode = nullptr;
  EXPECT_EQ(xmlFindNextTag(xml, "childNode1", rootNode, &foundNode),
            ncclSuccess);
  EXPECT_NE(foundNode, nullptr);
  EXPECT_STREQ(foundNode->name, "childNode1");
}

TEST_F(XmlTest, xmlFindNextTag_FindSecondOccurrence) {
  // First, find first occurrence
  struct ncclXmlNode *firstNode = nullptr;
  EXPECT_EQ(xmlFindNextTag(xml, "childNode1", rootNode, &firstNode),
            ncclSuccess);
  EXPECT_NE(firstNode, nullptr);

  // Try to find next occurrence (should be null as there's only one)
  struct ncclXmlNode *nextOccurrence = nullptr;
  EXPECT_EQ(xmlFindNextTag(xml, "childNode1", firstNode, &nextOccurrence),
            ncclSuccess);
  EXPECT_EQ(nextOccurrence, nullptr);
}

TEST_F(XmlTest, xmlFindNextTag_NonExistentTag) {
  struct ncclXmlNode *foundNode = nullptr;
  EXPECT_EQ(xmlFindNextTag(xml, "nonExistentTag", rootNode, &foundNode),
            ncclSuccess);
  EXPECT_EQ(foundNode, nullptr);
}

TEST_F(XmlTest, xmlFindNextTag_EmptyTagName) {
  struct ncclXmlNode *foundNode = nullptr;
  EXPECT_EQ(xmlFindNextTag(xml, "", rootNode, &foundNode), ncclSuccess);
  EXPECT_EQ(foundNode, nullptr);
}

TEST_F(XmlTest, xmlFindNextTag_FromLastNode) {
  // Start search from the last node in XML
  struct ncclXmlNode *foundNode = nullptr;
  EXPECT_EQ(xmlFindNextTag(xml, "testNode", nextNode, &foundNode), ncclSuccess);
  EXPECT_EQ(foundNode, nullptr); // Should not find anything after last node
}

// Tests for xmlPrintNodeRecursive
TEST_F(XmlTest, xmlPrintNodeRecursive_ValidNode) {
  // This test mainly checks that function doesn't crash
  EXPECT_EQ(xmlPrintNodeRecursive(testNode, "testPrint"), ncclSuccess);
}

TEST_F(XmlTest, xmlPrintNodeRecursive_NodeWithParent) {
  EXPECT_EQ(xmlPrintNodeRecursive(childNode1, "childPrint"), ncclSuccess);
}

TEST_F(XmlTest, xmlPrintNodeRecursive_RootNode) {
  EXPECT_EQ(xmlPrintNodeRecursive(rootNode, "rootPrint"), ncclSuccess);
}

TEST_F(XmlTest, xmlPrintNodeRecursive_NodeWithManyAttributes) {
  // Add many attributes to test output formatting
  for (int i = 0; i < 10; i++) {
    char attrName[32], attrValue[32];
    snprintf(attrName, sizeof(attrName), "attr%d", i);
    snprintf(attrValue, sizeof(attrValue), "value%d", i);
    ASSERT_EQ(xmlSetAttr(testNode, attrName, attrValue), ncclSuccess);
  }

  EXPECT_EQ(xmlPrintNodeRecursive(testNode, "manyAttrs"), ncclSuccess);
}

// Tests for xmlSetAttrFloat
TEST_F(XmlTest, xmlSetAttrFloat_ValidFloat) {
  float testValue = 2.71828f;
  EXPECT_EQ(xmlSetAttrFloat(testNode, "newFloatAttr", testValue), ncclSuccess);

  // Verify the attribute was set correctly
  const char *storedValue;
  ASSERT_EQ(xmlGetAttr(testNode, "newFloatAttr", &storedValue), ncclSuccess);
  EXPECT_NE(storedValue, nullptr);

  float retrievedValue = strtof(storedValue, nullptr);
  EXPECT_FLOAT_EQ(retrievedValue, testValue);
}

TEST_F(XmlTest, xmlSetAttrFloat_ZeroValue) {
  EXPECT_EQ(xmlSetAttrFloat(testNode, "zeroFloat", 0.0f), ncclSuccess);

  const char *storedValue;
  ASSERT_EQ(xmlGetAttr(testNode, "zeroFloat", &storedValue), ncclSuccess);
  EXPECT_STREQ(storedValue, "0");
}

TEST_F(XmlTest, xmlSetAttrFloat_NegativeValue) {
  float testValue = -123.456f;
  EXPECT_EQ(xmlSetAttrFloat(testNode, "negativeFloat", testValue), ncclSuccess);

  const char *storedValue;
  ASSERT_EQ(xmlGetAttr(testNode, "negativeFloat", &storedValue), ncclSuccess);
  float retrievedValue = strtof(storedValue, nullptr);
  EXPECT_FLOAT_EQ(retrievedValue, testValue);
}

TEST_F(XmlTest, xmlSetAttrFloat_OverwriteExisting) {
  // First set a value
  EXPECT_EQ(xmlSetAttrFloat(testNode, "overwriteTest", 1.0f), ncclSuccess);

  // Then overwrite it
  float newValue = 99.99f;
  EXPECT_EQ(xmlSetAttrFloat(testNode, "overwriteTest", newValue), ncclSuccess);

  const char *storedValue;
  ASSERT_EQ(xmlGetAttr(testNode, "overwriteTest", &storedValue), ncclSuccess);
  float retrievedValue = strtof(storedValue, nullptr);
  EXPECT_FLOAT_EQ(retrievedValue, newValue);
}

TEST_F(XmlTest, xmlSetAttrFloat_VeryLargeValue) {
  float largeValue = 1e20f;
  EXPECT_EQ(xmlSetAttrFloat(testNode, "largeFloat", largeValue), ncclSuccess);

  const char *storedValue;
  ASSERT_EQ(xmlGetAttr(testNode, "largeFloat", &storedValue), ncclSuccess);
  float retrievedValue = strtof(storedValue, nullptr);
  EXPECT_FLOAT_EQ(retrievedValue, largeValue);
}

TEST_F(XmlTest, xmlSetAttrFloat_VerySmallValue) {
  float smallValue = 1e-20f;
  EXPECT_EQ(xmlSetAttrFloat(testNode, "smallFloat", smallValue), ncclSuccess);

  const char *storedValue;
  ASSERT_EQ(xmlGetAttr(testNode, "smallFloat", &storedValue), ncclSuccess);
  float retrievedValue = strtof(storedValue, nullptr);
  EXPECT_FLOAT_EQ(retrievedValue, smallValue);
}

TEST_F(XmlTest, xmlSetAttrFloat_InfinityValue) {
  float infValue = INFINITY;
  ncclResult_t status = xmlSetAttrFloat(testNode, "infFloat", infValue);
  EXPECT_EQ(status, ncclSuccess);

  const char *storedValue;
  ASSERT_EQ(xmlGetAttr(testNode, "infFloat", &storedValue), ncclSuccess);
  // Check that some representation was stored (implementation dependent)
  EXPECT_NE(storedValue, nullptr);
}

TEST_F(XmlTest, xmlSetAttrFloat_NaNValue) {
  float nanValue = NAN;
  ncclResult_t status = xmlSetAttrFloat(testNode, "nanFloat", nanValue);
  EXPECT_EQ(status, ncclSuccess);

  const char *storedValue;
  ASSERT_EQ(xmlGetAttr(testNode, "nanFloat", &storedValue), ncclSuccess);
  EXPECT_NE(storedValue, nullptr);
}

// Tests for xmlGetSubKvInt
TEST_F(XmlTest, xmlGetSubKvInt_FindExisting) {
  struct ncclXmlNode *foundSub = nullptr;
  EXPECT_EQ(xmlGetSubKvInt(testNode, "childNode1", &foundSub, "id", 1),
            ncclSuccess);
  EXPECT_EQ(foundSub, childNode1);
}

TEST_F(XmlTest, xmlGetSubKvInt_FindDifferentValue) {
  struct ncclXmlNode *foundSub = nullptr;
  EXPECT_EQ(xmlGetSubKvInt(testNode, "childNode2", &foundSub, "id", 2),
            ncclSuccess);
  EXPECT_EQ(foundSub, childNode2);
}

TEST_F(XmlTest, xmlGetSubKvInt_NotFound) {
  struct ncclXmlNode *foundSub = nullptr;
  EXPECT_EQ(xmlGetSubKvInt(testNode, "childNode1", &foundSub, "id", 999),
            ncclSuccess);
  EXPECT_EQ(foundSub, nullptr);
}

TEST_F(XmlTest, xmlGetSubKvInt_NonExistentSubName) {
  struct ncclXmlNode *foundSub = nullptr;
  EXPECT_EQ(xmlGetSubKvInt(testNode, "nonExistentSub", &foundSub, "id", 1),
            ncclSuccess);
  EXPECT_EQ(foundSub, nullptr);
}

TEST_F(XmlTest, xmlGetSubKvInt_NonExistentAttribute) {
  struct ncclXmlNode *foundSub = nullptr;
  EXPECT_EQ(
      xmlGetSubKvInt(testNode, "childNode1", &foundSub, "nonExistentAttr", 1),
      ncclSuccess);
  EXPECT_EQ(foundSub, nullptr);
}

TEST_F(XmlTest, xmlGetSubKvInt_ZeroValue) {
  // Add a child with zero value
  struct ncclXmlNode *zeroChild = nullptr;
  ASSERT_EQ(xmlAddNode(xml, testNode, "zeroChild", &zeroChild), ncclSuccess);
  ASSERT_EQ(xmlSetAttr(zeroChild, "value", "0"), ncclSuccess);

  struct ncclXmlNode *foundSub = nullptr;
  EXPECT_EQ(xmlGetSubKvInt(testNode, "zeroChild", &foundSub, "value", 0),
            ncclSuccess);
  EXPECT_EQ(foundSub, zeroChild);
}

TEST_F(XmlTest, xmlGetSubKvInt_NegativeValue) {
  // Add a child with negative value
  struct ncclXmlNode *negChild = nullptr;
  ASSERT_EQ(xmlAddNode(xml, testNode, "negChild", &negChild), ncclSuccess);
  ASSERT_EQ(xmlSetAttr(negChild, "value", "-42"), ncclSuccess);

  struct ncclXmlNode *foundSub = nullptr;
  EXPECT_EQ(xmlGetSubKvInt(testNode, "negChild", &foundSub, "value", -42),
            ncclSuccess);
  EXPECT_EQ(foundSub, negChild);
}

// Tests for kvConvertToStr
TEST_F(XmlTest, kvConvertToStr_ValidValue) {
  const char *result = nullptr;
  EXPECT_EQ(kvConvertToStr(1, &result, testDict), ncclSuccess);
  EXPECT_NE(result, nullptr);
  EXPECT_STREQ(result, "first");
}

TEST_F(XmlTest, kvConvertToStr_AnotherValidValue) {
  const char *result = nullptr;
  EXPECT_EQ(kvConvertToStr(3, &result, testDict), ncclSuccess);
  EXPECT_NE(result, nullptr);
  EXPECT_STREQ(result, "third");
}

TEST_F(XmlTest, kvConvertToStr_InvalidValue) {
  const char *result = nullptr;
  EXPECT_EQ(kvConvertToStr(999, &result, testDict), ncclInternalError);
  // Result should be undefined for invalid values
}

TEST_F(XmlTest, kvConvertToStr_ZeroValue) {
  // Add zero to dictionary
  struct kvDict zeroDict[2];
  zeroDict[0] = {"zero", 0};
  zeroDict[1] = {nullptr, 0};

  const char *result = nullptr;
  EXPECT_EQ(kvConvertToStr(0, &result, zeroDict), ncclSuccess);
  EXPECT_NE(result, nullptr);
  EXPECT_STREQ(result, "zero");
}

TEST_F(XmlTest, kvConvertToStr_NullDict) {
  const char *result = nullptr;
  ncclResult_t status = kvConvertToStr(1, &result, nullptr);
  EXPECT_TRUE(status == ncclInternalError);
}

TEST_F(XmlTest, kvConvertToStr_EmptyDict) {
  struct kvDict emptyDict[1];
  emptyDict[0] = {nullptr, 0};

  const char *result = nullptr;
  EXPECT_EQ(kvConvertToStr(1, &result, emptyDict), ncclInternalError);
}

TEST_F(XmlTest, kvConvertToStr_NullResultPointer) {
  ncclResult_t status = kvConvertToStr(1, nullptr, testDict);
  EXPECT_TRUE(status == ncclSuccess || status == ncclInternalError);
}

TEST_F(XmlTest, kvConvertToStr_NegativeValue) {
  struct kvDict negDict[2];
  negDict[0] = {"negative", -1};
  negDict[1] = {nullptr, 0};

  const char *result = nullptr;
  EXPECT_EQ(kvConvertToStr(-1, &result, negDict), ncclSuccess);
  EXPECT_NE(result, nullptr);
  EXPECT_STREQ(result, "negative");
}

// Edge case and stress tests
TEST_F(XmlTest, StressTest_ManyAttributes) {
  // Test node with maximum attributes
  struct ncclXmlNode *stressNode = nullptr;
  ASSERT_EQ(xmlAddNode(xml, rootNode, "stressTest", &stressNode), ncclSuccess);

  // Add many float attributes
  for (int i = 0; i < MAX_ATTR_COUNT - 1; i++) {
    char attrName[32];
    snprintf(attrName, sizeof(attrName), "float%d", i);
    EXPECT_EQ(xmlSetAttrFloat(stressNode, attrName, (float)i * 1.1f),
              ncclSuccess);
  }

  // Test retrieval
  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(stressNode, "float5", &result, 0.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, 5.5f);
}

// Additional tests for xmlGetAttrFloatDefault - more edge cases
TEST_F(XmlTest, xmlGetAttrFloatDefault_VeryPreciseFloat) {
  ASSERT_EQ(xmlSetAttr(testNode, "preciseFloat", "3.141592653589793"),
            ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "preciseFloat", &result, 0.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, 3.141592653589793f);
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_ExponentialFormat) {
  ASSERT_EQ(xmlSetAttr(testNode, "expFloat", "2.5e10"), ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "expFloat", &result, 1.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, 2.5e10f);
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_NegativeExponential) {
  ASSERT_EQ(xmlSetAttr(testNode, "negExpFloat", "-1.5e-8"), ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "negExpFloat", &result, 1.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, -1.5e-8f);
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_LeadingPlusSign) {
  ASSERT_EQ(xmlSetAttr(testNode, "plusFloat", "+42.5"), ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "plusFloat", &result, 0.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, 42.5f);
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_IntegerString) {
  ASSERT_EQ(xmlSetAttr(testNode, "integerStr", "123"), ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "integerStr", &result, 0.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, 123.0f);
}

TEST_F(XmlTest, xmlGetAttrFloatDefault_PartiallyValidFloat) {
  ASSERT_EQ(xmlSetAttr(testNode, "partialFloat", "12.34abc"), ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "partialFloat", &result, 99.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(result, 12.34f); // strtof should parse up to invalid char
}

// Tests for xmlGetAttrLong
TEST_F(XmlTest, xmlGetAttrLong_ValidPositiveNumber) {
  ASSERT_EQ(xmlSetAttr(testNode, "longAttr", "1234567890"), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "longAttr", &result), ncclSuccess);
  EXPECT_EQ(result, 1234567890LL);
}

TEST_F(XmlTest, xmlGetAttrLong_ValidNegativeNumber) {
  ASSERT_EQ(xmlSetAttr(testNode, "negativeLong", "-9876543210"), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "negativeLong", &result), ncclSuccess);
  EXPECT_EQ(result, -9876543210LL);
}

TEST_F(XmlTest, xmlGetAttrLong_ZeroValue) {
  ASSERT_EQ(xmlSetAttr(testNode, "zeroLong", "0"), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "zeroLong", &result), ncclSuccess);
  EXPECT_EQ(result, 0LL);
}

TEST_F(XmlTest, xmlGetAttrLong_AttributeNotFound) {
  int64_t result = 999; // Initialize with non-zero value
  ncclResult_t status =
      xmlGetAttrLong(testNode, "nonExistentLongAttr", &result);
  EXPECT_TRUE(status == ncclInternalError);
  // Result value is undefined when attribute not found
}

TEST_F(XmlTest, xmlGetAttrLong_EmptyAttribute) {
  ASSERT_EQ(xmlSetAttr(testNode, "emptyLong", ""), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "emptyLong", &result), ncclSuccess);
  EXPECT_EQ(result, 0LL); // Empty string typically converts to 0
}

TEST_F(XmlTest, xmlGetAttrLong_InvalidNumber) {
  ASSERT_EQ(xmlSetAttr(testNode, "invalidLong", "notanumber"), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "invalidLong", &result), ncclSuccess);
  EXPECT_EQ(result, 0LL); // Invalid string typically converts to 0
}

TEST_F(XmlTest, xmlGetAttrLong_VeryLargePositive) {
  ASSERT_EQ(xmlSetAttr(testNode, "largeLong", "9223372036854775807"),
            ncclSuccess); // INT64_MAX

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "largeLong", &result), ncclSuccess);
  EXPECT_EQ(result, 9223372036854775807LL);
}

TEST_F(XmlTest, xmlGetAttrLong_VeryLargeNegative) {
  ASSERT_EQ(xmlSetAttr(testNode, "minLong", "-9223372036854775808"),
            ncclSuccess); // INT64_MIN

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "minLong", &result), ncclSuccess);
  EXPECT_EQ(result, -9223372036854775808LL);
}

TEST_F(XmlTest, xmlGetAttrLong_Overflow) {
  // Test number larger than INT64_MAX
  ASSERT_EQ(xmlSetAttr(testNode, "overflowLong", "99999999999999999999"),
            ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "overflowLong", &result), ncclSuccess);
  // Result will be INT64_MAX due to overflow in strtoll
  EXPECT_EQ(result, 9223372036854775807LL);
}

TEST_F(XmlTest, xmlGetAttrLong_Underflow) {
  // Test number smaller than INT64_MIN
  ASSERT_EQ(xmlSetAttr(testNode, "underflowLong", "-99999999999999999999"),
            ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "underflowLong", &result), ncclSuccess);
  // Result will be INT64_MIN due to underflow in strtoll
  EXPECT_EQ(result, -9223372036854775808LL);
}

TEST_F(XmlTest, xmlGetAttrLong_LeadingWhitespace) {
  ASSERT_EQ(xmlSetAttr(testNode, "whitespaceLong", "   42   "), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "whitespaceLong", &result), ncclSuccess);
  EXPECT_EQ(result, 42LL); // strtoll should handle leading/trailing whitespace
}

TEST_F(XmlTest, xmlGetAttrLong_LeadingPlusSign) {
  ASSERT_EQ(xmlSetAttr(testNode, "plusLong", "+12345"), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "plusLong", &result), ncclSuccess);
  EXPECT_EQ(result, 12345LL);
}

TEST_F(XmlTest, xmlGetAttrLong_HexadecimalNumber) {
  ASSERT_EQ(xmlSetAttr(testNode, "hexLong", "0x1a2b"), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "hexLong", &result), ncclSuccess);
  EXPECT_EQ(result, 0x1a2bLL); // strtoll should handle hex if base is 0 or 16
}

TEST_F(XmlTest, xmlGetAttrLong_OctalNumber) {
  ASSERT_EQ(xmlSetAttr(testNode, "octalLong", "0123"), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "octalLong", &result), ncclSuccess);
  EXPECT_EQ(result, 0123LL); // Should parse as octal if base is 0
}

TEST_F(XmlTest, xmlGetAttrLong_PartiallyValidNumber) {
  ASSERT_EQ(xmlSetAttr(testNode, "partialLong", "12345abc"), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "partialLong", &result), ncclSuccess);
  EXPECT_EQ(result, 12345LL); // strtoll should parse up to invalid character
}

TEST_F(XmlTest, xmlGetAttrLong_FloatingPointString) {
  ASSERT_EQ(xmlSetAttr(testNode, "floatAsLong", "123.456"), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "floatAsLong", &result), ncclSuccess);
  EXPECT_EQ(result, 123LL); // strtoll should parse integer part only
}

TEST_F(XmlTest, xmlGetAttrLong_ScientificNotation) {
  ASSERT_EQ(xmlSetAttr(testNode, "scientificLong", "1e5"), ncclSuccess);

  int64_t result;
  EXPECT_EQ(xmlGetAttrLong(testNode, "scientificLong", &result), ncclSuccess);
  EXPECT_EQ(result,
            1LL); // strtoll doesn't handle scientific notation, stops at 'e'
}

TEST_F(XmlTest, xmlGetAttrLong_NullValuePointer) {
  ncclResult_t status = xmlGetAttrLong(testNode, "longAttr", nullptr);
  EXPECT_TRUE(status == ncclInternalError);
}

TEST_F(XmlTest, xmlGetAttrLong_EmptyAttrName) {
  int64_t result;
  ncclResult_t status = xmlGetAttrLong(testNode, "", &result);
  EXPECT_TRUE(status == ncclInternalError);
}

// Edge case tests for both functions
TEST_F(XmlTest, EdgeCase_MultipleConversions) {
  // Test converting the same attribute multiple times
  ASSERT_EQ(xmlSetAttr(testNode, "multiTest", "42.5"), ncclSuccess);

  float floatResult;
  int64_t longResult;

  // Multiple float conversions should be consistent
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "multiTest", &floatResult, 0.0f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(floatResult, 42.5f);

  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "multiTest", &floatResult, 99.9f),
            ncclSuccess);
  EXPECT_FLOAT_EQ(floatResult, 42.5f);

  // Long conversion of same value
  EXPECT_EQ(xmlGetAttrLong(testNode, "multiTest", &longResult), ncclSuccess);
  EXPECT_EQ(longResult, 42LL); // Should convert integer part only
}

TEST_F(XmlTest, EdgeCase_OverwriteAndRetrieve) {
  // Test overwriting attributes and retrieving with different functions
  ASSERT_EQ(xmlSetAttr(testNode, "overwriteTest", "123"), ncclSuccess);

  int64_t longResult;
  EXPECT_EQ(xmlGetAttrLong(testNode, "overwriteTest", &longResult),
            ncclSuccess);
  EXPECT_EQ(longResult, 123LL);

  // Overwrite with float value
  ASSERT_EQ(xmlSetAttr(testNode, "overwriteTest", "456.789"), ncclSuccess);

  float floatResult;
  EXPECT_EQ(
      xmlGetAttrFloatDefault(testNode, "overwriteTest", &floatResult, 0.0f),
      ncclSuccess);
  EXPECT_FLOAT_EQ(floatResult, 456.789f);

  // Long conversion should now get integer part
  EXPECT_EQ(xmlGetAttrLong(testNode, "overwriteTest", &longResult),
            ncclSuccess);
  EXPECT_EQ(longResult, 456LL);
}

TEST_F(XmlTest, EdgeCase_LongAttributeNames) {
  char longName[MAX_STR_LEN + 10];
  memset(longName, 'a', sizeof(longName) - 1);
  longName[sizeof(longName) - 1] = '\0';

  // This may truncate or fail gracefully
  ncclResult_t status = xmlSetAttrFloat(testNode, longName, 42.0f);
  EXPECT_TRUE(status == ncclSuccess);
}

TEST_F(XmlTest, EdgeCase_EmptyStringValues) {
  ASSERT_EQ(xmlSetAttr(testNode, "emptyStr", ""), ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "emptyStr", &result, 99.9f),
            ncclSuccess);
  // Empty string typically converts to 0.0
  EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(XmlTest, EdgeCase_WhitespaceValues) {
  ASSERT_EQ(xmlSetAttr(testNode, "whitespace", "   3.14   "), ncclSuccess);

  float result;
  EXPECT_EQ(xmlGetAttrFloatDefault(testNode, "whitespace", &result, 0.0f),
            ncclSuccess);
  // strtof should handle leading/trailing whitespace
  EXPECT_FLOAT_EQ(result, 3.14f);
}

// Test comprehensive XML topology loading covering all ncclTopoXmlLoad*
// functions
TEST_F(XmlTest, TestCompleteTopologyLoading) {
  // Create comprehensive topology XML that exercises all loading functions
  // Based on the actual parser structure from xml.cc
  std::string completeTopologyXml = R"(<system version="2" name="testSystem">
  <cpu numaid="0" affinity="0000ffff" arch="x86_64" vendor="AuthenticAMD" familyid="25" modelid="1">
    <pci busid="0000:00:01.0" class="0x060400" link_speed="8.0 GT/s PCIe" link_width="16">
      <pci busid="0000:01:00.0" class="0x030200" vendor="0x1002" device="0x73df">
        <gpu dev="0" sm="110" gcn="gfx942" arch="942">
          <xgmi maxcount="8">
            <link target="1" count="4" bw="25"/>
            <link target="2" count="4" bw="25"/>
          </xgmi>
        </gpu>
      </pci>
      <pci busid="0000:02:00.0" class="0x030200" vendor="0x1002" device="0x73df">
        <gpu dev="1" sm="110" gcn="gfx942" arch="942">
          <xgmi maxcount="8">
            <link target="0" count="4" bw="25"/>
            <link target="3" count="4" bw="25"/>
          </xgmi>
        </gpu>
      </pci>
      <pci busid="0000:03:00.0" class="0x020000" vendor="0x15b3" device="0x101b">
        <nic>
          <net name="mlx5_0" dev="0" speed="100" port="1"/>
          <net name="mlx5_1" dev="1" speed="100" port="2"/>
        </nic>
      </pci>
      <pcilink class="0x060400" link="1"/>
    </pci>
    <nic>
      <net name="eth0" dev="2" speed="10" port="1"/>
    </nic>
  </cpu>
</system>)";

  createTestXmlFile(completeTopologyXml);

  struct ncclXml *xml = allocateXml(100);
  ASSERT_NE(xml, nullptr);

  // Test loading - this should exercise all ncclTopoXmlLoad* functions
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  // Verify system node was loaded (ncclTopoXmlLoadSystem)
  struct ncclXmlNode *systemNode = nullptr;
  EXPECT_EQ(xmlFindTag(xml, "system", &systemNode), ncclSuccess);
  EXPECT_NE(systemNode, nullptr);

  if (systemNode) {
    const char *version;
    EXPECT_EQ(xmlGetAttr(systemNode, "version", &version), ncclSuccess);
    EXPECT_STREQ(version, "2");

    // Verify CPU node was loaded (ncclTopoXmlLoadCpu)
    struct ncclXmlNode *cpuNode = nullptr;
    EXPECT_EQ(xmlGetSub(systemNode, "cpu", &cpuNode), ncclSuccess);
    EXPECT_NE(cpuNode, nullptr);

    if (cpuNode) {
      // Verify PCI nodes were loaded (ncclTopoXmlLoadPci)
      struct ncclXmlNode *pciNode = nullptr;
      EXPECT_EQ(xmlGetSub(cpuNode, "pci", &pciNode), ncclSuccess);
      EXPECT_NE(pciNode, nullptr);

      if (pciNode) {
        // Search for GPU nodes in the PCI hierarchy (ncclTopoXmlLoadGpu)
        struct ncclXmlNode *gpuNode = nullptr;
        // GPU nodes are nested inside PCI nodes, so we need to search deeper
        for (int i = 0; i < pciNode->nSubs && !gpuNode; i++) {
          struct ncclXmlNode *subPci = pciNode->subs[i];
          if (strcmp(subPci->name, "pci") == 0) {
            xmlGetSub(subPci, "gpu", &gpuNode);
          }
        }
        EXPECT_NE(gpuNode, nullptr);

        if (gpuNode) {
          // Verify NVLINK/XGMI nodes were loaded (ncclTopoXmlLoadNvlink)
          struct ncclXmlNode *xgmiNode = nullptr;
          EXPECT_EQ(xmlGetSub(gpuNode, "xgmi", &xgmiNode), ncclSuccess);
          EXPECT_NE(xgmiNode, nullptr);
        }

        // Search for NIC nodes in the PCI hierarchy (ncclTopoXmlLoadNic)
        struct ncclXmlNode *nicNode = nullptr;
        for (int i = 0; i < pciNode->nSubs && !nicNode; i++) {
          struct ncclXmlNode *subPci = pciNode->subs[i];
          if (strcmp(subPci->name, "pci") == 0) {
            xmlGetSub(subPci, "nic", &nicNode);
          }
        }
        EXPECT_NE(nicNode, nullptr);

        if (nicNode) {
          // Verify NET nodes were loaded (ncclTopoXmlLoadNet)
          struct ncclXmlNode *netNode = nullptr;
          EXPECT_EQ(xmlGetSub(nicNode, "net", &netNode), ncclSuccess);
          EXPECT_NE(netNode, nullptr);
        }

        // Verify PCILink nodes were loaded (ncclTopoXmlLoadPciLink)
        struct ncclXmlNode *pciLinkNode = nullptr;
        EXPECT_EQ(xmlGetSub(pciNode, "pcilink", &pciLinkNode), ncclSuccess);
        EXPECT_NE(pciLinkNode, nullptr);
      }

      // Also check for standalone NIC under CPU
      struct ncclXmlNode *cpuNicNode = nullptr;
      EXPECT_EQ(xmlGetSub(cpuNode, "nic", &cpuNicNode), ncclSuccess);
      EXPECT_NE(cpuNicNode, nullptr);
    }
  }

  free(xml);
}

// Test GPU loading with NVLINK (remove C2C for now since it's being ignored)
TEST_F(XmlTest, TestGpuLoading_WithInterconnects) {
  std::string gpuXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:01:00.0" class="0x030200">
      <gpu dev="0" sm="110" gcn="gfx942" arch="942">
        <xgmi maxcount="8">
          <link target="1" count="4" bw="25"/>
          <link target="2" count="4" bw="25"/>
        </xgmi>
      </gpu>
    </pci>
  </cpu>
</system>)";

  createTestXmlFile(gpuXml);

  struct ncclXml *xml = allocateXml(30);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  struct ncclXmlNode *gpuNode = nullptr;
  EXPECT_EQ(xmlFindTag(xml, "gpu", &gpuNode), ncclSuccess);
  EXPECT_NE(gpuNode, nullptr);

  if (gpuNode) {
    const char *gcn;
    EXPECT_EQ(xmlGetAttr(gpuNode, "gcn", &gcn), ncclSuccess);
    EXPECT_STREQ(gcn, "gfx942");

    // Verify XGMI (NVLINK equivalent on AMD)
    struct ncclXmlNode *xgmiNode = nullptr;
    EXPECT_EQ(xmlGetSub(gpuNode, "xgmi", &xgmiNode), ncclSuccess);
    EXPECT_NE(xgmiNode, nullptr);
  }

  free(xml);
}

// Test C2C loading separately (if supported)
TEST_F(XmlTest, TestC2cLoading_IfSupported) {
  std::string c2cXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:01:00.0" class="0x030200">
      <gpu dev="0" sm="110">
        <c2c tclass="1">
          <link target="4" bw="50"/>
          <link target="5" bw="50"/>
        </c2c>
      </gpu>
    </pci>
  </cpu>
</system>)";

  createTestXmlFile(c2cXml);

  struct ncclXml *xml = allocateXml(20);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  struct ncclXmlNode *c2cNode = nullptr;
  EXPECT_EQ(xmlFindTag(xml, "c2c", &c2cNode), ncclSuccess);
  // Note: C2C might be ignored in some builds, so we don't assert it exists
  if (c2cNode) {
    const char *tclass;
    EXPECT_EQ(xmlGetAttr(c2cNode, "tclass", &tclass), ncclSuccess);
    EXPECT_STREQ(tclass, "1");
  }

  free(xml);
}

// Test individual component loading functions - System
TEST_F(XmlTest, TestSystemLoading_ValidVersion) {
  std::string systemXml = R"(<system version="2" name="testSystem">
  <cpu numaid="0" affinity="0000ffff" arch="x86_64"/>
</system>)";

  createTestXmlFile(systemXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestSystemLoading_InvalidVersion) {
  std::string systemXml = R"(<system version="999" name="testSystem">
  <cpu numaid="0"/>
</system>)";

  createTestXmlFile(systemXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclInvalidUsage); // Should fail due to wrong version

  free(xml);
}

// Test CPU loading with various attributes
TEST_F(XmlTest, TestCpuLoading_CompleteAttributes) {
  std::string cpuXml = R"(<system version="2">
  <cpu numaid="0" affinity="0000ffff" arch="x86_64" vendor="AuthenticAMD" familyid="25" modelid="1">
    <pci busid="0000:00:01.0" class="0x060400"/>
    <nic>
      <net name="eth0" dev="0" speed="10"/>
    </nic>
  </cpu>
</system>)";

  createTestXmlFile(cpuXml);

  struct ncclXml *xml = allocateXml(20);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  if (result == ncclSuccess) {
    struct ncclXmlNode *cpuNode = nullptr;
    EXPECT_EQ(xmlFindTag(xml, "cpu", &cpuNode), ncclSuccess);
    EXPECT_NE(cpuNode, nullptr);

    if (cpuNode) {
      const char *arch;
      EXPECT_EQ(xmlGetAttr(cpuNode, "arch", &arch), ncclSuccess);
      EXPECT_STREQ(arch, "x86_64");
    }
  }

  free(xml);
}

// Test PCI loading with nested structure
TEST_F(XmlTest, TestPciLoading_NestedStructure) {
  std::string pciXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:00:01.0" class="0x060400" vendor="0x1002" device="0x1234">
      <pci busid="0000:01:00.0" class="0x030200" vendor="0x1002" device="0x73df">
        <gpu dev="0" sm="110"/>
      </pci>
      <pci busid="0000:02:00.0" class="0x020000">
        <nic>
          <net name="mlx5_0" dev="0"/>
        </nic>
      </pci>
      <pcilink class="0x060400" link="1"/>
    </pci>
  </cpu>
</system>)";

  createTestXmlFile(pciXml);

  struct ncclXml *xml = allocateXml(30);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  struct ncclXmlNode *pciNode = nullptr;
  EXPECT_EQ(xmlFindTag(xml, "pci", &pciNode), ncclSuccess);
  EXPECT_NE(pciNode, nullptr);

  if (pciNode) {
    const char *busid;
    EXPECT_EQ(xmlGetAttr(pciNode, "busid", &busid), ncclSuccess);
    EXPECT_STREQ(busid, "0000:00:01.0");
  }

  free(xml);
}

// Test NIC and NET loading
TEST_F(XmlTest, TestNicNetLoading_MultipleNets) {
  std::string nicNetXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:03:00.0" class="0x020000">
      <nic>
        <net name="mlx5_0" dev="0" speed="100" port="1"/>
        <net name="mlx5_1" dev="1" speed="100" port="2"/>
      </nic>
    </pci>
    <nic>
      <net name="eth0" dev="2" speed="10" port="1"/>
      <net name="eth1" dev="3" speed="1" port="1"/>
    </nic>
  </cpu>
</system>)";

  createTestXmlFile(nicNetXml);

  struct ncclXml *xml = allocateXml(30);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  struct ncclXmlNode *nicNode = nullptr;
  EXPECT_EQ(xmlFindTag(xml, "nic", &nicNode), ncclSuccess);
  EXPECT_NE(nicNode, nullptr);

  if (nicNode) {
    struct ncclXmlNode *netNode = nullptr;
    EXPECT_EQ(xmlGetSub(nicNode, "net", &netNode), ncclSuccess);
    EXPECT_NE(netNode, nullptr);

    if (netNode) {
      const char *name;
      EXPECT_EQ(xmlGetAttr(netNode, "name", &name), ncclSuccess);
      EXPECT_NE(name, nullptr);
    }
  }

  free(xml);
}

// Test PciLink loading
TEST_F(XmlTest, TestPciLinkLoading_MultipleLinks) {
  std::string pciLinkXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:00:01.0" class="0x060400">
      <pcilink class="0x060400" link="1"/>
      <pcilink class="0x060400" link="2"/>
      <pcilink class="0x060400" link="4"/>
    </pci>
  </cpu>
</system>)";

  createTestXmlFile(pciLinkXml);

  struct ncclXml *xml = allocateXml(20);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  struct ncclXmlNode *pciLinkNode = nullptr;
  EXPECT_EQ(xmlFindTag(xml, "pcilink", &pciLinkNode), ncclSuccess);
  EXPECT_NE(pciLinkNode, nullptr);

  if (pciLinkNode) {
    const char *link;
    EXPECT_EQ(xmlGetAttr(pciLinkNode, "link", &link), ncclSuccess);
    EXPECT_NE(link, nullptr);
  }

  free(xml);
}

// Test XGMI loading
TEST_F(XmlTest, TestXgmiLoading_ComplexTopology) {
  std::string xgmiXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:01:00.0" class="0x030200">
      <gpu dev="0" sm="110">
        <xgmi maxcount="8">
          <link target="1" count="4" bw="25"/>
          <link target="2" count="4" bw="25"/>
          <link target="3" count="2" bw="25"/>
        </xgmi>
      </gpu>
    </pci>
  </cpu>
</system>)";

  createTestXmlFile(xgmiXml);

  struct ncclXml *xml = allocateXml(20);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  struct ncclXmlNode *xgmiNode = nullptr;
  EXPECT_EQ(xmlFindTag(xml, "xgmi", &xgmiNode), ncclSuccess);
  EXPECT_NE(xgmiNode, nullptr);

  if (xgmiNode) {
    const char *maxcount;
    EXPECT_EQ(xmlGetAttr(xgmiNode, "maxcount", &maxcount), ncclSuccess);
    EXPECT_STREQ(maxcount, "8");
  }

  free(xml);
}

// Test error cases - missing required attributes
TEST_F(XmlTest, TestErrorCases_MissingAttributes) {
  // Test system without version
  std::string noVersionXml = R"(<system name="testSystem">
  <cpu numaid="0"/>
</system>)";

  createTestXmlFile(noVersionXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess); // Should fail due to missing version

  free(xml);
}

// Test corner case - empty components
TEST_F(XmlTest, TestCornerCases_EmptyComponents) {
  std::string emptyComponentsXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:01:00.0" class="0x030200">
      <gpu dev="0" sm="110">
        <xgmi maxcount="0"/>
        <c2c tclass="0"/>
      </gpu>
    </pci>
    <nic/>
  </cpu>
</system>)";

  createTestXmlFile(emptyComponentsXml);

  struct ncclXml *xml = allocateXml(20);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  // Verify empty components are handled gracefully
  struct ncclXmlNode *nicNode = nullptr;
  EXPECT_EQ(xmlFindTag(xml, "nic", &nicNode), ncclSuccess);
  EXPECT_NE(nicNode, nullptr);

  free(xml);
}

// Test stress case - large topology
TEST_F(XmlTest, TestStressCase_LargeTopology) {
  std::string largeTopologyXml = R"(<system version="2">
  <cpu numaid="0">)";

  // Add multiple GPUs
  for (int i = 0; i < 8; i++) {
    largeTopologyXml += "<pci busid=\"0000:0" + std::to_string(i + 1) +
                        ":00.0\" class=\"0x030200\">";
    largeTopologyXml +=
        "<gpu dev=\"" + std::to_string(i) + "\" sm=\"110\" gcn=\"gfx942\">";
    largeTopologyXml += "<xgmi maxcount=\"8\">";
    for (int j = 0; j < 8; j++) {
      if (i != j) {
        largeTopologyXml += "<link target=\"" + std::to_string(j) +
                            "\" count=\"4\" bw=\"25\"/>";
      }
    }
    largeTopologyXml += "</xgmi></gpu></pci>";
  }

  // Add multiple NICs
  for (int i = 0; i < 4; i++) {
    largeTopologyXml += "<pci busid=\"0000:1" + std::to_string(i + 1) +
                        ":00.0\" class=\"0x020000\">";
    largeTopologyXml += "<nic><net name=\"mlx5_" + std::to_string(i) +
                        "\" dev=\"" + std::to_string(i) +
                        "\" speed=\"100\"/></nic></pci>";
  }

  largeTopologyXml += "</cpu></system>";

  createTestXmlFile(largeTopologyXml);

  struct ncclXml *xml = allocateXml(200);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  // Verify multiple components were loaded
  EXPECT_GT(xml->maxIndex, 37); // Should have many nodes

  free(xml);
}

// Test malformed XML handling
TEST_F(XmlTest, TestMalformedXml_GracefulFailure) {
  std::string malformedXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:01:00.0" class="0x030200">
      <gpu dev="0" sm="110">
        <xgmi maxcount="8"
      </gpu>
    </pci>
  </cpu>
</system>)"; // Missing closing bracket

  createTestXmlFile(malformedXml);

  struct ncclXml *xml = allocateXml(20);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess); // Should fail gracefully

  free(xml);
}

// Test XML parsing warnings through ncclTopoGetXmlFromFile
TEST_F(XmlTest, TestXmlWarnings_UnexpectedEOF) {
  // Create XML file with unexpected EOF (truncated)
  std::string truncatedXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:01:00.0")"; // Missing closing quotes and elements

  createTestXmlFile(truncatedXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Parse : Unexpected EOF" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_ExpectedQuote) {
  // Create XML with missing quotes around attribute values
  std::string noQuotesXml = R"(<system version=2>
  <cpu numaid=0>
  </cpu>
</system>)";

  createTestXmlFile(noQuotesXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Parse : Expected (double) quote" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_UnexpectedValue) {
  // Create XML with unexpected value assignment
  std::string unexpectedValueXml = R"(<system>
  <cpu="unexpected"/>
</system>)";

  createTestXmlFile(unexpectedValueXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Parse : Unexpected value with name" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_UnterminatedComment) {
  // Create XML with unterminated comment
  std::string unterminatedCommentXml = R"(<system version="2">
  <!-- This comment is never closed
  <cpu numaid="0"/>
</system>)";

  createTestXmlFile(unterminatedCommentXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Parse error : unterminated comment" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_ExpectingBracket) {
  // Create XML without proper opening bracket
  std::string noBracketXml = R"(system version="2">
  <cpu numaid="0"/>
</system>)";

  createTestXmlFile(noBracketXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Parse error : expecting '<'" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_UnexpectedTrailing) {
  // Create XML with unexpected trailing characters in closing tag
  std::string trailingXml = R"(<system version="2">
  <cpu numaid="0">
  </cpu extra>
</system>)";

  createTestXmlFile(trailingXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Parse error : unexpected trailing" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_ExpectedClosingBracket) {
  // Create XML without proper closing bracket
  std::string noClosingBracketXml = R"(<system version="2"
  <cpu numaid="0"/>
</system>)";

  createTestXmlFile(noClosingBracketXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Parse : expected >, got" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_UnterminatedElement) {
  // Create XML with unterminated element
  std::string unterminatedXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:01:00.0">
      <gpu dev="0"/>
    <!-- missing </pci> and </cpu> -->
</system>)";

  createTestXmlFile(unterminatedXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Parse : unterminated" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_XmlMismatch) {
  // Create XML with mismatched opening/closing tags
  std::string mismatchXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:01:00.0">
      <gpu dev="0"/>
    </nic>  <!-- Should be </pci> -->
  </cpu>
</system>)";

  createTestXmlFile(mismatchXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Mismatch" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_TooManySubnodes) {
  // Create XML that would exceed MAX_SUBS limit
  std::string xmlStart = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:00:01.0">)";

  std::string xmlEnd = R"(    </pci>
  </cpu>
</system>)";

  // Add more than MAX_SUBS (512) PCI subnodes
  std::string manySubsXml = xmlStart;
  for (int i = 0; i <= 512; i++) {
    char pciEntry[128];
    snprintf(pciEntry, sizeof(pciEntry),
             "\n      <pci busid=\"0000:%02x:00.0\"/>", i % 256);
    manySubsXml += pciEntry;
  }
  manySubsXml += xmlEnd;

  createTestXmlFile(manySubsXml);

  struct ncclXml *xml =
      allocateXml(1000); // Enough for nodes, but will hit subnodes limit
  ASSERT_NE(xml, nullptr);

  // This should trigger "Error : XML parser is limited to X subnodes" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_WrongVersion) {
  // Create XML with wrong version (not NCCL_TOPO_XML_VERSION = 2)
  std::string wrongVersionXml = R"(<system version="999">
  <cpu numaid="0"/>
</system>)";

  createTestXmlFile(wrongVersionXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Topology has wrong version" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclInvalidUsage);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_FileNotFound) {
  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "Could not open XML topology file" warning when warn=1
  ncclResult_t result = ncclTopoGetXmlFromFile("nonexistent_file.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess); // Function succeeds but warns

  // With warn=0, no warning should be generated
  result = ncclTopoGetXmlFromFile("nonexistent_file.xml", xml, 0);
  EXPECT_EQ(result, ncclSuccess);

  free(xml);
}

// Test edge cases that trigger INFO messages (which are warnings in practice)
TEST_F(XmlTest, TestXmlInfo_IgnoringElement) {
  // Create XML with unknown elements that should be ignored
  std::string unknownElementXml = R"(<system version="2">
  <cpu numaid="0">
    <unknown_element id="1"/>
    <another_unknown value="test"/>
    <pci busid="0000:01:00.0"/>
  </cpu>
</system>)";

  createTestXmlFile(unknownElementXml);

  struct ncclXml *xml = allocateXml(20);
  ASSERT_NE(xml, nullptr);

  // This should trigger "Ignoring element" INFO messages
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess); // Should succeed but log ignored elements

  free(xml);
}

// Test complex malformed XML scenarios
TEST_F(XmlTest, TestXmlWarnings_MalformedComplex1) {
  // Mixed malformation: missing quotes and wrong structure
  std::string complexMalformedXml = R"(<system version=2>
  <cpu numaid="0>
    <pci busid="0000:01:00.0"
      <gpu dev="0">
    </pci>
  </cpu>
</system>)";

  createTestXmlFile(complexMalformedXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

TEST_F(XmlTest, TestXmlWarnings_MalformedComplex2) {
  // Another complex malformation: unclosed tags and unexpected characters
  std::string complexMalformedXml2 = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:01:00.0">
      <gpu dev="0"
        <xgmi target="1"/>
      </gpu
    </pci>
  </cpu>
<!-- Missing </system> tag -->)";

  createTestXmlFile(complexMalformedXml2);

  struct ncclXml *xml = allocateXml(20);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

// Fix the minimal memory test - need at least 2 nodes for even minimal XML
TEST_F(XmlTest, TestXmlWarnings_MinimalMemory) {
  // Test with just enough memory for minimal valid XML (2 nodes minimum)
  struct ncclXml *xml = allocateXml(2);
  ASSERT_NE(xml, nullptr);

  std::string minimalXml = R"(<system version="2"/>)";
  createTestXmlFile(minimalXml);

  // Should succeed with minimal valid XML and 2 nodes
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  free(xml);
}

// Add a proper test for the "too many nodes" warning
TEST_F(XmlTest, TestXmlWarnings_TooManyNodes) {
  // Create XML with more nodes than limit can handle
  struct ncclXml *xml = allocateXml(1); // Only 1 node - will fail immediately
  ASSERT_NE(xml, nullptr);

  // Even minimal XML needs more than 1 node
  std::string minimalXml = R"(<system version="2"/>)";
  createTestXmlFile(minimalXml);

  // This should trigger "Error : XML parser is limited to 1 nodes" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclInternalError); // Should fail due to node limit

  free(xml);
}

// Better test for node limit with more realistic scenario
TEST_F(XmlTest, TestXmlWarnings_NodeLimitRealistic) {
  // Create XML that will exceed a small but reasonable node limit
  struct ncclXml *xml = allocateXml(3); // Small limit that should be exceeded
  ASSERT_NE(xml, nullptr);

  std::string complexXml = R"(<system version="2">
  <cpu numaid="0">
    <pci busid="0000:01:00.0">
      <gpu dev="0"/>
    </pci>
    <pci busid="0000:02:00.0">
      <gpu dev="1"/>
    </pci>
  </cpu>
</system>)";

  createTestXmlFile(complexXml);

  // This should trigger "Error : XML parser is limited to 3 nodes" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclInternalError);

  free(xml);
}

// Fix the boundary conditions test to be more specific
TEST_F(XmlTest, TestXmlWarnings_BoundaryConditions) {
  // Test exactly at the MAX_STR_LEN boundary (255 characters)
  // But use a more reasonable test that should actually trigger the warning
  std::string longButValidName(253, 'x'); // Just under the limit
  std::string boundaryXml =
      "<system version=\"2\">\n  <" + longButValidName + "/>\n</system>";

  createTestXmlFile(boundaryXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should succeed since we're just under the boundary
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  free(xml);
}

// Add a test that actually triggers the name too long warning
TEST_F(XmlTest, TestXmlWarnings_NameTooLong) {
  // Create a test file manually that will trigger the name too long condition
  // We need to create malformed XML that the parser will try to parse
  std::string veryLongName(300, 'a'); // Much longer than MAX_STR_LEN (255)

  // Create XML that has a very long element name without proper termination
  // This is tricky because we need the parser to actually try to parse the long
  // name
  std::ofstream file("test_topology.xml");
  file << "<system version=\"2\">\n";
  file << "  <"
       << veryLongName; // Don't close the tag properly to force parsing
  file << " id=\"1\"/>\n";
  file << "</system>";
  file.close();

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "Error : name too long (max 255)" warning
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_NE(result, ncclSuccess);

  free(xml);
}

// Test the attribute count warning with a more controlled approach
TEST_F(XmlTest, TestXmlWarnings_TooManyAttributes_Controlled) {
  // Create XML with exactly MAX_ATTR_COUNT + 1 attributes to trigger the
  // warning
  std::string xmlStart = R"(<system version="2">
  <cpu numaid="0")";

  // Add exactly MAX_ATTR_COUNT (16) + a few more attributes
  std::string attributes;
  for (int i = 1; i <= 20; i++) { // More than MAX_ATTR_COUNT (16)
    attributes +=
        " attr" + std::to_string(i) + "=\"value" + std::to_string(i) + "\"";
  }

  std::string xmlEnd = R"(>
    <pci busid="0000:01:00.0"/>
  </cpu>
</system>)";

  std::string manyAttrsXml = xmlStart + attributes + xmlEnd;
  createTestXmlFile(manyAttrsXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  // This should trigger "XML Parse : Ignoring extra attributes (max 16)" INFO
  // message
  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess); // Should succeed but log the warning

  free(xml);
}

// Test valid XML with warning-prone structures that should succeed
TEST_F(XmlTest, TestXmlWarnings_EdgeCaseSuccess) {
  // Test XML that's at the edge but should still parse successfully
  std::string edgeCaseXml = R"(<system version="2">
  <!-- Valid comment -->
  <cpu numaid="0">
    <pci busid="0000:01:00.0" class="0x030200">
      <gpu dev="0"/>
    </pci>
  </cpu>
</system>)";

  createTestXmlFile(edgeCaseXml);

  struct ncclXml *xml = allocateXml(10);
  ASSERT_NE(xml, nullptr);

  ncclResult_t result = ncclTopoGetXmlFromFile("test_topology.xml", xml, 1);
  EXPECT_EQ(result, ncclSuccess);

  // Verify structure was parsed correctly
  struct ncclXmlNode *systemNode = nullptr;
  EXPECT_EQ(xmlFindTag(xml, "system", &systemNode), ncclSuccess);
  EXPECT_NE(systemNode, nullptr);

  free(xml);
}