/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/xml.cc and xml.h: the topology/graph XML parser and writer, and the
// XML autodetection from sysfs, HIP and amd-smi, all served by fakes/graph_fakes.cc.

#include <gtest/gtest.h>

#include <cpuid.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ScopedHook.h"
#include "fakes/graph_fakes.h"
#include "fakes/param_redirect.h"

#include XML_CC_PATH

#include "TopoScene.h"

using namespace topo_scene;

namespace {

using Attrs = std::vector<std::pair<std::string, std::string>>;

class XmlMicrotest : public ::testing::Test {
 protected:
  XmlPtr xml_ = NewXml();

  void SetUp() override { ResetGraphFakes(); }
  void TearDown() override { ResetGraphFakes(); }

  ncclResult_t Parse(const std::string& text) { return ParseXml(text, xml_.get()); }

  ncclXmlNode* Add(ncclXmlNode* parent, const char* name, const Attrs& attrs = {}) {
    ncclXmlNode* node = nullptr;
    EXPECT_EQ(ncclSuccess, xmlAddNode(xml_.get(), parent, name, &node));
    for (const auto& kv : attrs) EXPECT_EQ(ncclSuccess, xmlSetAttr(node, kv.first.c_str(), kv.second.c_str()));
    return node;
  }

  std::string Dump() {
    TempFile file;
    EXPECT_EQ(ncclSuccess, ncclTopoDumpXmlToFile(file.path(), xml_.get()));
    return file.Read();
  }

  ncclXmlNode* Find(const char* tag, const char* key, const char* value) {
    ncclXmlNode* node = nullptr;
    EXPECT_EQ(ncclSuccess, xmlFindTagKv(xml_.get(), tag, &node, key, value));
    return node;
  }

  static std::string Attr(ncclXmlNode* node, const char* key) {
    const char* value = nullptr;
    EXPECT_EQ(ncclSuccess, xmlGetAttr(node, key, &value));
    return value ? value : "<unset>";
  }
};

// --- Parser

TEST_F(XmlMicrotest, GetXmlFromFile_BuildsTheNodeTreeWithItsAttributes) {
  ASSERT_EQ(ncclSuccess, Parse("<!-- topology -->\n"
                               "<system version='2' name=\"box\">\n"
                               "  <cpu numaid=\"0\">\n"
                               "    <pci busid=\"0000:01:00.0\">\n"
                               "      <pcilink target=\"0000:02:00.0\"/>\n"
                               "      <gpu dev=\"0\"><xgmi target=\"0000:03:00.0\" count=\"2\"/></gpu>\n"
                               "      <nic><net name=\"a\"/><gin name=\"b\"/><rma name=\"c\"/></nic>\n"
                               "    </pci>\n"
                               "    <nic><net name=\"d\"/></nic>\n"
                               "  </cpu>\n"
                               "</system>\n"));

  ASSERT_EQ(12, xml_->maxIndex);
  ncclXmlNode* system = xml_->nodes;
  EXPECT_STREQ("system", system->name);
  EXPECT_EQ(nullptr, system->parent);
  EXPECT_EQ("box", Attr(system, "name"));
  ncclXmlNode* pci = Find("pci", "busid", "0000:01:00.0");
  ASSERT_NE(nullptr, pci);
  EXPECT_STREQ("cpu", pci->parent->name);
  EXPECT_EQ(3, pci->nSubs);
  ncclXmlNode* xgmi = Find("xgmi", "target", "0000:03:00.0");
  ASSERT_NE(nullptr, xgmi);
  EXPECT_EQ("2", Attr(xgmi, "count"));
  EXPECT_EQ(NODE_TYPE_SINGLE, xgmi->type);
  EXPECT_STREQ("gpu", xgmi->parent->name);
  EXPECT_NE(nullptr, Find("rma", "name", "c"));
  EXPECT_STREQ("nic", Find("net", "name", "d")->parent->name);
  EXPECT_TRUE(Logged("Loading topology box"));
}

TEST_F(XmlMicrotest, GetXmlFromFile_SkipsUnknownElementsWithTheirChildren) {
  ASSERT_EQ(ncclSuccess, Parse("<system version=\"2\"><foo a=\"1\"><bar/><baz></baz></foo>"
                               "<cpu numaid=\"0\"/></system>"));

  EXPECT_EQ(2, xml_->maxIndex);
  EXPECT_STREQ("cpu", xml_->nodes[1].name);
  EXPECT_EQ(1, xml_->nodes[0].nSubs);
  EXPECT_TRUE(Logged("Ignoring element foo"));
  EXPECT_FALSE(Logged("Ignoring element bar"));
  EXPECT_TRUE(Logged("Loading unnamed topology"));
}

TEST_F(XmlMicrotest, GetXmlFromFile_CommentsMayEndInsideTheTagName) {
  ASSERT_EQ(ncclSuccess, Parse("<!--x--><!-- a > b -- c --><system version=\"2\"/>"));

  EXPECT_EQ(1, xml_->maxIndex);
}

TEST_F(XmlMicrotest, GetXmlFromFile_ExtraAttributesAreReadButDropped) {
  std::string attrs;
  for (int a = 0; a < MAX_ATTR_COUNT + 2; a++) attrs += " k" + std::to_string(a) + "=\"" + std::to_string(a) + "\"";

  ASSERT_EQ(ncclSuccess, Parse("<system version=\"2\"><cpu" + attrs + "/></system>"));

  ncclXmlNode* cpu = xml_->nodes + 1;
  EXPECT_EQ(MAX_ATTR_COUNT, cpu->nAttrs);
  EXPECT_EQ("15", Attr(cpu, "k15"));
  EXPECT_EQ("<unset>", Attr(cpu, "k16"));
  EXPECT_TRUE(Logged("Ignoring extra attributes"));
}

TEST_F(XmlMicrotest, GetXmlFromFile_MissingFileIsNotAnError) {
  xml_->maxIndex = 7;

  EXPECT_EQ(ncclSuccess, ncclTopoGetXmlFromFile("/nonexistent/topo.xml", xml_.get(), 0));
  EXPECT_FALSE(Logged("Could not open XML topology file"));
  EXPECT_EQ(ncclSuccess, ncclTopoGetXmlFromFile("/nonexistent/topo.xml", xml_.get(), 1));

  EXPECT_TRUE(Logged("Could not open XML topology file /nonexistent/topo.xml"));
  EXPECT_EQ(7, xml_->maxIndex);
}

TEST_F(XmlMicrotest, GetXmlFromFile_WrongVersionIsInvalidUsage) {
  EXPECT_EQ(ncclInvalidUsage, Parse("<system version=\"1\"></system>"));

  EXPECT_TRUE(Logged("XML Topology has wrong version 1, 2 needed"));
}

TEST_F(XmlMicrotest, GetXmlFromFile_SystemWithoutVersionFails) {
  EXPECT_EQ(ncclInternalError, Parse("<system></system>"));

  EXPECT_TRUE(Logged("Attribute version of node system not found"));
}

struct BadXml {
  const char* text;
  const char* warn;
};

TEST_F(XmlMicrotest, GetXmlFromFile_MalformedInputFailsWithTheParserWarning) {
  const std::string longValue(MAX_STR_LEN, 'v');
  const std::string longName(MAX_STR_LEN, 'n');
  const std::string valueText = "<system version=\"" + longValue + "\"/>";
  const std::string nameText = "<" + longName + "/>";
  const BadXml cases[] = {
    {"x<system version=\"2\"/>", "expecting '<', got 'x'"},
    {"<system version=2/>", "Expected (double) quote"},
    {"<system version=\"2", "Unexpected EOF"},
    {"<system version=\"2\"><cpu></system>", "XML Mismatch : cpu / system"},
    {"<system version=\"2\"><cpu>", "unterminated cpu"},
    {"<system version=\"2\"></system x>", "unexpected trailing   in closing tag system"},
    {"<system version=\"2\"c>", "expected >, got 'c'"},
    {"<system version=\"2\"><cpu\n/></system>", "expected >, got '\n'"},
    {"<system=\"2\">", "Unexpected value with name system"},
    {"<!-- never closed", "unterminated comment"},
    {valueText.c_str(), "too long (max 255)"},
    {nameText.c_str(), "too long (max 255)"},
  };
  for (const BadXml& c : cases) {
    g_logs.clear();

    EXPECT_EQ(ncclInternalError, Parse(c.text)) << c.text;

    EXPECT_TRUE(Logged(c.warn)) << c.text;
  }
}

TEST_F(XmlMicrotest, GetXmlFromFile_StopsAtMaxNodes) {
  xml_ = NewXml(2);

  EXPECT_EQ(ncclInternalError, Parse("<system version=\"2\"><cpu/><cpu/></system>"));

  EXPECT_TRUE(Logged("XML parser is limited to 2 nodes"));
}

TEST_F(XmlMicrotest, GetXmlFromFile_StopsAtMaxSubnodes) {
  xml_ = NewXml(MAX_SUBS + 4);
  std::string cpus;
  for (int s = 0; s <= MAX_SUBS; s++) cpus += "<cpu/>";

  EXPECT_EQ(ncclInternalError, Parse("<system version=\"2\">" + cpus + "</system>"));

  EXPECT_TRUE(Logged("XML parser is limited to 512 subnodes"));
}

// The gpu handler only knows xgmi on AMD, so nothing routes to the C2C loader.
TEST_F(XmlMicrotest, LoadC2c_ConsumesItsSubtree) {
  TempFile file("<a/></c2c>");
  FILE* f = fopen(file.path(), "r");
  ncclXmlNode* c2c = Add(nullptr, "c2c");
  c2c->type = NODE_TYPE_OPEN;

  EXPECT_EQ(ncclSuccess, ncclTopoXmlLoadC2c(f, xml_.get(), c2c));

  fclose(f);
  EXPECT_EQ(1, xml_->maxIndex);
}

// --- Writer

TEST_F(XmlMicrotest, DumpXmlToFile_IndentsChildrenAndSelfClosesLeaves) {
  ncclXmlNode* system = Add(nullptr, "system", {{"version", "2"}});
  ncclXmlNode* cpu = Add(system, "cpu", {{"numaid", "0"}});
  Add(cpu, "pci", {{"busid", "0000:01:00.0"}, {"class", "0x03"}});

  EXPECT_EQ("<system version=\"2\">\n"
            "  <cpu numaid=\"0\">\n"
            "    <pci busid=\"0000:01:00.0\" class=\"0x03\"/>\n"
            "  </cpu>\n"
            "</system>\n",
            Dump());
}

TEST_F(XmlMicrotest, DumpXmlToFile_ParsesBackToTheSameText) {
  ASSERT_EQ(ncclSuccess, Parse(XgmiSystem(2, 1, true)));
  std::string first = Dump();

  ASSERT_EQ(ncclSuccess, Parse(first));

  EXPECT_EQ(first, Dump());
}

TEST_F(XmlMicrotest, DumpXmlToFile_UnwritablePathIsIgnored) {
  Add(nullptr, "system");

  EXPECT_EQ(ncclSuccess, ncclTopoDumpXmlToFile("/nonexistent/dir/topo.xml", xml_.get()));

  EXPECT_TRUE(Logged("Unable to open /nonexistent/dir/topo.xml, not dumping topology"));
}

// --- Serialization and fusion

TEST_F(XmlMicrotest, ConvertXml_RoundTripsPointersThroughOffsets) {
  ncclXmlNode* system = Add(nullptr, "system");
  ncclXmlNode* cpu = Add(system, "cpu");
  Add(cpu, "pci");
  XmlPtr copy = NewXml();
  memcpy(copy.get(), xml_.get(), xmlMemSize(kXmlNodes));

  ASSERT_EQ(ncclSuccess, ncclTopoConvertXml(copy.get(), (uintptr_t)xml_->nodes, 1));

  EXPECT_EQ(nullptr, copy->nodes[0].parent);
  EXPECT_EQ((uintptr_t)1, (uintptr_t)copy->nodes[1].parent);  // shifted by one so the first node is not NULL
  EXPECT_EQ((uintptr_t)sizeof(ncclXmlNode), (uintptr_t)copy->nodes[0].subs[0]);

  ASSERT_EQ(ncclSuccess, ncclTopoConvertXml(copy.get(), (uintptr_t)copy->nodes, 0));

  EXPECT_EQ(nullptr, copy->nodes[0].parent);
  EXPECT_EQ(copy->nodes + 0, copy->nodes[1].parent);
  EXPECT_EQ(copy->nodes + 1, copy->nodes[0].subs[0]);
  EXPECT_EQ(copy->nodes + 2, copy->nodes[1].subs[0]);
}

TEST_F(XmlMicrotest, FuseXml_IntoEmptyXmlCopiesTheWholeTree) {
  XmlPtr src = MustParseXml(XgmiSystem(2));

  ASSERT_EQ(ncclSuccess, ncclTopoFuseXml(xml_.get(), src.get()));

  EXPECT_EQ(src->maxIndex, xml_->maxIndex);
  EXPECT_NE(nullptr, Find("gpu", "dev", "1"));
}

TEST_F(XmlMicrotest, FuseXml_MergesEqualNodesAndAppendsTheRest) {
  ASSERT_EQ(ncclSuccess, Parse("<system version=\"2\"><cpu numaid=\"0\"><pci busid=\"a\"/></cpu></system>"));
  XmlPtr src = MustParseXml("<system version=\"2\"><cpu numaid=\"0\"><pci busid=\"b\"/></cpu>"
                            "<cpu numaid=\"1\"/></system>");

  ASSERT_EQ(ncclSuccess, ncclTopoFuseXml(xml_.get(), src.get()));

  ncclXmlNode* cpu0 = Find("cpu", "numaid", "0");
  ASSERT_NE(nullptr, cpu0);
  ASSERT_EQ(2, cpu0->nSubs);
  EXPECT_EQ("a", Attr(cpu0->subs[0], "busid"));
  EXPECT_EQ("b", Attr(cpu0->subs[1], "busid"));
  EXPECT_EQ(2, xml_->nodes[0].nSubs);
  EXPECT_EQ(5, xml_->maxIndex);
}

TEST_F(XmlMicrotest, FindNode_NeedsTheSameNameTypeAndEveryAttribute) {
  ASSERT_EQ(ncclSuccess, Parse("<system version=\"2\"><cpu numaid=\"0\" arch=\"x\"/><cpu numaid=\"1\"/></system>"));
  XmlPtr src = MustParseXml("<system version=\"2\"><cpu numaid=\"0\" arch=\"y\"/><cpu numaid=\"1\"></cpu>"
                            "<cpu numaid=\"0\" arch=\"x\"/></system>");
  ncclXmlNode* found = nullptr;

  EXPECT_EQ(ncclSuccess, xmlFindNode(xml_->nodes, src->nodes[0].subs[0], &found));
  EXPECT_EQ(nullptr, found);
  EXPECT_EQ(ncclSuccess, xmlFindNode(xml_->nodes, src->nodes[0].subs[1], &found));
  EXPECT_EQ(nullptr, found);  // open/close, not self-closing
  EXPECT_EQ(ncclSuccess, xmlFindNode(xml_->nodes, src->nodes[0].subs[2], &found));
  EXPECT_EQ(xml_->nodes + 1, found);
}

// --- Graph files

TEST_F(XmlMicrotest, GetXmlGraphFromFile_LoadsChannelsWithTheirEndpoints) {
  TempFile file("<graphs version=\"1\" name=\"g\"><graph id=\"0\"><channel><net dev=\"0\"/><gpu dev=\"0\"/>"
                "<gpu dev=\"1\"/><net dev=\"0\"/></channel><extra/></graph></graphs>");

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlGraphFromFile(file.path(), xml_.get()));

  EXPECT_EQ(7, xml_->maxIndex);
  EXPECT_EQ(4, xml_->nodes[2].nSubs);
  EXPECT_TRUE(Logged("Loading graphs for topology g"));
}

TEST_F(XmlMicrotest, GetXmlGraphFromFile_UnnamedGraphsLoad) {
  TempFile file("<graphs version=\"1\"/>");

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlGraphFromFile(file.path(), xml_.get()));

  EXPECT_TRUE(Logged("Loading graphs"));
  EXPECT_FALSE(Logged("Loading graphs for"));
}

TEST_F(XmlMicrotest, GetXmlGraphFromFile_MissingFileIsSystemError) {
  EXPECT_EQ(ncclSystemError, ncclTopoGetXmlGraphFromFile("/nonexistent/graph.xml", xml_.get()));

  EXPECT_TRUE(Logged("Could not open XML graph file /nonexistent/graph.xml"));
}

TEST_F(XmlMicrotest, GetXmlGraphFromFile_WrongVersionIsInvalidUsage) {
  TempFile file("<graphs version=\"2\"/>");

  EXPECT_EQ(ncclInvalidUsage, ncclTopoGetXmlGraphFromFile(file.path(), xml_.get()));

  EXPECT_TRUE(Logged("XML Graph has wrong version 2, 1 needed"));
}

// --- Trimming

TEST_F(XmlMicrotest, TrimXml_DropsBranchesWithoutAKeptNode) {
  ASSERT_EQ(ncclSuccess,
            Parse("<system version=\"2\">"
                  "<cpu numaid=\"0\"><pci busid=\"a\"><gpu dev=\"0\" keep=\"1\"/><gpu dev=\"1\"/></pci>"
                  "<pci busid=\"b\"><nic><net name=\"n0\" keep=\"1\"/><gin name=\"g0\"/><rma name=\"r0\"/></nic></pci>"
                  "<pci busid=\"c\" keep=\"0\"/></cpu>"
                  "<cpu numaid=\"1\"><pci busid=\"c\"/></cpu></system>"));

  ASSERT_EQ(ncclSuccess, ncclTopoTrimXml(xml_.get()));

  ncclXmlNode* system = xml_->nodes;
  ASSERT_EQ(1, system->nSubs);
  ncclXmlNode* cpu = system->subs[0];
  EXPECT_EQ("0", Attr(cpu, "numaid"));
  ASSERT_EQ(2, cpu->nSubs);
  ncclXmlNode* pci = cpu->subs[0];
  EXPECT_EQ("a", Attr(pci, "busid"));
  ASSERT_EQ(2, pci->nSubs);  // only pci/cpu/nic/net/gin/rma are trimmed, so the unkept gpu stays
  EXPECT_EQ("<unset>", Attr(pci->subs[0], "keep"));
  ncclXmlNode* nic = cpu->subs[1]->subs[0];
  ASSERT_EQ(1, nic->nSubs);
  EXPECT_STREQ("net", nic->subs[0]->name);
}

TEST_F(XmlMicrotest, UnsetAttr_ShiftsTheLaterAttributesDown) {
  ncclXmlNode* node = Add(nullptr, "n", {{"a", "1"}, {"b", "2"}, {"c", "3"}});

  ASSERT_EQ(ncclSuccess, xmlUnsetAttr(node, "b"));
  ASSERT_EQ(ncclSuccess, xmlUnsetAttr(node, "missing"));

  ASSERT_EQ(2, node->nAttrs);
  EXPECT_STREQ("c", node->attrs[1].key);
  EXPECT_STREQ("3", node->attrs[1].value);
}

// --- xml.h accessors

TEST_F(XmlMicrotest, AttrGetters_ParseNumbersAndFallBackOnlyInTheDefaultVariants) {
  ncclXmlNode* node =
    Add(nullptr, "n", {{"i", "0x10"}, {"l", "-5"}, {"u", "0xffffffffffffffff"}, {"f", "1.5"}, {"s", "str"}});
  int i = 0;
  int64_t l = 0;
  uint64_t u = 0;
  float f = 0;

  EXPECT_EQ(ncclSuccess, xmlGetAttrInt(node, "i", &i));
  EXPECT_EQ(16, i);
  EXPECT_EQ(ncclSuccess, xmlGetAttrLong(node, "l", &l));
  EXPECT_EQ(-5, l);
  EXPECT_EQ(ncclSuccess, xmlGetAttrUint64(node, "u", &u));
  EXPECT_EQ(UINT64_MAX, u);
  EXPECT_EQ(ncclSuccess, xmlGetAttrFloat(node, "f", &f));
  EXPECT_FLOAT_EQ(1.5f, f);
  EXPECT_EQ(ncclSuccess, xmlGetAttrIntDefault(node, "x", &i, 7));
  EXPECT_EQ(7, i);
  EXPECT_EQ(ncclSuccess, xmlGetAttrIntDefault(node, "i", &i, 7));
  EXPECT_EQ(16, i);
  EXPECT_EQ(ncclSuccess, xmlGetAttrUint64Default(node, "x", &u, 9));
  EXPECT_EQ(9u, u);
  EXPECT_EQ(ncclSuccess, xmlGetAttrUint64Default(node, "u", &u, 9));
  EXPECT_EQ(UINT64_MAX, u);
  EXPECT_EQ(ncclSuccess, xmlGetAttrFloatDefault(node, "x", &f, 2.5f));
  EXPECT_FLOAT_EQ(2.5f, f);
  EXPECT_EQ(ncclSuccess, xmlGetAttrFloatDefault(node, "f", &f, 2.5f));
  EXPECT_FLOAT_EQ(1.5f, f);
  EXPECT_EQ(ncclInternalError, xmlGetAttrInt(node, "x", &i));
  EXPECT_EQ(ncclInternalError, xmlGetAttrLong(node, "x", &l));
  EXPECT_EQ(ncclInternalError, xmlGetAttrUint64(node, "x", &u));
  EXPECT_EQ(ncclInternalError, xmlGetAttrFloat(node, "x", &f));
  EXPECT_TRUE(Logged("Attribute x of node n not found"));
  EXPECT_FALSE(Logged("HINT"));
}

TEST_F(XmlMicrotest, AttrGetStr_MissingNicBusIdAddsTheContainerHint) {
  ncclXmlNode* nic = Add(nullptr, "nic");
  const char* value = nullptr;

  EXPECT_EQ(ncclInternalError, xmlGetAttrStr(nic, "busid", &value));

  EXPECT_TRUE(Logged("'--net host'"));
}

TEST_F(XmlMicrotest, AttrSetters_FormatTheirTypeAndOverwriteInPlace) {
  ncclXmlNode* node = Add(nullptr, "n");

  ASSERT_EQ(ncclSuccess, xmlSetAttrInt(node, "i", -3));
  ASSERT_EQ(ncclSuccess, xmlSetAttrFloat(node, "f", 0.25f));
  ASSERT_EQ(ncclSuccess, xmlSetAttrLong(node, "l", 0x1234));
  ASSERT_EQ(ncclSuccess, xmlSetAttrUint64(node, "u", 0xabc));
  ASSERT_EQ(ncclSuccess, xmlSetAttrIfUnset(node, "s", "first"));
  ASSERT_EQ(ncclSuccess, xmlSetAttrIfUnset(node, "s", "second"));
  ASSERT_EQ(ncclSuccess, xmlSetAttrInt(node, "i", 4));
  ASSERT_EQ(ncclSuccess, xmlSetAttrFloat(node, "f", 2));
  ASSERT_EQ(ncclSuccess, xmlSetAttrLong(node, "l", 0x10));
  ASSERT_EQ(ncclSuccess, xmlSetAttrUint64(node, "u", 0));

  EXPECT_EQ(5, node->nAttrs);
  EXPECT_EQ("4", Attr(node, "i"));
  EXPECT_EQ("2", Attr(node, "f"));
  EXPECT_EQ("0x10", Attr(node, "l"));
  EXPECT_EQ("0x0", Attr(node, "u"));
  EXPECT_EQ("first", Attr(node, "s"));
}

TEST_F(XmlMicrotest, AttrSetters_FailOnceTheNodeIsFull) {
  ncclXmlNode* node = Add(nullptr, "n");
  for (int a = 0; a < MAX_ATTR_COUNT; a++) ASSERT_EQ(ncclSuccess, xmlSetAttrInt(node, std::to_string(a).c_str(), a));

  EXPECT_EQ(ncclInternalError, xmlSetAttr(node, "x", "1"));
  EXPECT_EQ(ncclInternalError, xmlSetAttrIfUnset(node, "x", "1"));
  EXPECT_EQ(ncclInternalError, xmlSetAttrInt(node, "x", 1));
  EXPECT_EQ(ncclInternalError, xmlSetAttrFloat(node, "x", 1));
  EXPECT_EQ(ncclInternalError, xmlSetAttrLong(node, "x", 1));
  EXPECT_EQ(ncclInternalError, xmlSetAttrUint64(node, "x", 1));

  EXPECT_EQ(MAX_ATTR_COUNT, node->nAttrs);
  EXPECT_TRUE(Logged("too many XML attributes (max 16)"));
}

TEST_F(XmlMicrotest, FindAndGetSub_MatchByNameAndAttribute) {
  ncclXmlNode* system = Add(nullptr, "system");
  ncclXmlNode* cpu0 = Add(system, "cpu", {{"numaid", "0"}});
  ncclXmlNode* pci = Add(system, "pci", {{"busid", "x"}});
  ncclXmlNode* cpu7 = Add(system, "cpu", {{"numaid", "7"}});
  ncclXmlNode* node = nullptr;

  EXPECT_EQ(ncclSuccess, xmlFindTag(xml_.get(), "cpu", &node));
  EXPECT_EQ(cpu0, node);
  EXPECT_EQ(ncclSuccess, xmlFindNextTag(xml_.get(), "cpu", node, &node));
  EXPECT_EQ(cpu7, node);
  EXPECT_EQ(ncclSuccess, xmlFindNextTag(xml_.get(), "cpu", node, &node));
  EXPECT_EQ(nullptr, node);
  EXPECT_EQ(ncclSuccess, xmlFindTag(xml_.get(), "gpu", &node));
  EXPECT_EQ(nullptr, node);
  EXPECT_EQ(nullptr, Find("cpu", "numaid", "8"));
  EXPECT_EQ(ncclSuccess, xmlGetSub(system, "pci", &node));
  EXPECT_EQ(pci, node);
  EXPECT_EQ(ncclSuccess, xmlGetSub(system, "nic", &node));
  EXPECT_EQ(nullptr, node);
  EXPECT_EQ(ncclSuccess, xmlGetSubKvInt(system, "cpu", &node, "numaid", 7));
  EXPECT_EQ(cpu7, node);
  EXPECT_EQ(ncclSuccess, xmlGetSubKv(system, "cpu", &node, "numaid", "1"));
  EXPECT_EQ(nullptr, node);
}

TEST_F(XmlMicrotest, AddNode_FailsPastMaxNodesAndMaxSubs) {
  xml_ = NewXml(MAX_SUBS + 2);
  ncclXmlNode* parent = Add(nullptr, "p");
  ncclXmlNode* node = nullptr;
  for (int s = 0; s < MAX_SUBS; s++) ASSERT_EQ(ncclSuccess, xmlAddNode(xml_.get(), parent, "c", &node));

  EXPECT_EQ(ncclInternalError, xmlAddNode(xml_.get(), parent, "c", &node));
  EXPECT_TRUE(Logged("too many XML subnodes (max 512)"));
  EXPECT_EQ(ncclInternalError, xmlAddNode(xml_.get(), parent, "c", &node));
  EXPECT_TRUE(Logged("too many XML nodes (max 514)"));
}

TEST_F(XmlMicrotest, AddTree_CopiesRecursivelyAndFailsWhenFull) {
  XmlPtr src = MustParseXml(XgmiSystem(2));
  ASSERT_EQ(ncclSuccess, xmlAddTree(xml_.get(), nullptr, src->nodes));
  EXPECT_EQ(src->maxIndex, xml_->maxIndex);

  XmlPtr small = NewXml(3);
  EXPECT_EQ(ncclInternalError, xmlAddTree(small.get(), nullptr, src->nodes));
  EXPECT_TRUE(Logged("too many XML nodes (max 3)"));

  XmlPtr wide = NewXml(MAX_SUBS + 2);
  ncclXmlNode* parent = nullptr;
  ncclXmlNode* sub = nullptr;
  ASSERT_EQ(ncclSuccess, xmlAddNode(wide.get(), nullptr, "p", &parent));
  for (int s = 0; s < MAX_SUBS; s++) ASSERT_EQ(ncclSuccess, xmlAddNode(wide.get(), parent, "c", &sub));
  EXPECT_EQ(ncclInternalError, xmlAddTree(wide.get(), parent, sub));
  EXPECT_TRUE(Logged("too many XML subnodes (max 512)"));
}

TEST_F(XmlMicrotest, RemoveNode_UnlinksFromTheParentAndMarksTheNodeUnused) {
  ncclXmlNode* parent = Add(nullptr, "p");
  ncclXmlNode* a = Add(parent, "a");
  ncclXmlNode* b = Add(parent, "b");
  ncclXmlNode* c = Add(parent, "c");

  ASSERT_EQ(ncclSuccess, xmlRemoveNode(b));
  ASSERT_EQ(ncclSuccess, xmlRemoveNode(parent));

  EXPECT_EQ(NODE_TYPE_NONE, b->type);
  ASSERT_EQ(2, parent->nSubs);
  EXPECT_EQ(a, parent->subs[0]);
  EXPECT_EQ(c, parent->subs[1]);
}

TEST_F(XmlMicrotest, KvConvert_PrefixMatchesWithFallbackToTheTerminator) {
  struct kvDict dict[] = {{"0x03", 1}, {"0x02", 2}, {NULL, 9}};
  int value = 0;
  const char* str = nullptr;

  EXPECT_EQ(ncclSuccess, kvConvertToInt("0x030000", &value, dict));
  EXPECT_EQ(1, value);
  EXPECT_EQ(ncclSuccess, kvConvertToInt("0x12", &value, dict));
  EXPECT_EQ(9, value);
  EXPECT_TRUE(Logged("could not find value of '0x12' in dictionary, falling back to 9"));
  EXPECT_EQ(ncclSuccess, kvConvertToStr(2, &str, dict));
  EXPECT_STREQ("0x02", str);
  EXPECT_EQ(ncclInternalError, kvConvertToStr(5, &str, dict));
  EXPECT_TRUE(Logged("could not find value 5 in dictionary"));
  EXPECT_EQ(ncclInternalError, kvConvertToStr(1, &str, nullptr));
  EXPECT_TRUE(Logged("null dictionary"));
  EXPECT_EQ(ncclInternalError, kvConvertToStr(1, nullptr, dict));
  EXPECT_TRUE(Logged("null result pointer"));
}

TEST_F(XmlMicrotest, PrintNodeRecursive_LogsTheChainUpToTheRoot) {
  ncclXmlNode* system = Add(nullptr, "system");
  ncclXmlNode* gpu = Add(system, "gpu", {{"dev", "3"}});

  EXPECT_EQ(ncclSuccess, xmlPrintNodeRecursive(gpu, "gpu"));

  EXPECT_TRUE(Logged("<name=gpu dev=3>"));
  EXPECT_TRUE(Logged("<name=system>"));
}

TEST_F(XmlMicrotest, CheckBDFFormat_NeedsTwelveHexCharsWithSeparators) {
  char good[] = "0000:aB:0f.7";
  char shortId[] = "0000:00:00";
  char longId[] = "0000:00:00.00";
  char badSep[] = "0000-00:00.0";
  char badDigit[] = "0000:0g:00.0";
  char badFunction[] = "0000:00:00.x";

  EXPECT_EQ(1, checkBDFFormat(good));
  EXPECT_EQ(0, checkBDFFormat(shortId));
  EXPECT_EQ(0, checkBDFFormat(longId));
  EXPECT_EQ(0, checkBDFFormat(badSep));
  EXPECT_EQ(0, checkBDFFormat(badDigit));
  EXPECT_EQ(0, checkBDFFormat(badFunction));
  EXPECT_EQ(1, isHex('F'));
  EXPECT_EQ(0, isHex('G'));
}

// --- Autodetection: CPU and sysfs

struct CpuId {
  std::string vendor;
  int family;
  int model;
};

CpuId HostCpuId() {
  unsigned eax, ebx, ecx, edx;
  __cpuid(0, eax, ebx, ecx, edx);
  char vendor[13];
  memcpy(vendor, &ebx, 4);
  memcpy(vendor + 4, &edx, 4);
  memcpy(vendor + 8, &ecx, 4);
  vendor[12] = '\0';
  __cpuid(1, eax, ebx, ecx, edx);
  int family = (eax >> 8) & 0xf, model = (eax >> 4) & 0xf;
  if (family == 15 || family == 6) model += ((eax >> 16) & 0xf) << 4;
  if (family == 15) family += (eax >> 20) & 0xff;
  return {vendor, family, model};
}

class XmlSysfsMicrotest : public XmlMicrotest {
 protected:
  ncclXmlNode* system_ = nullptr;

  void SetUp() override {
    XmlMicrotest::SetUp();
    system_ = Add(nullptr, "system", {{"version", "2"}});
  }

  // A PCI function at `path`, with the sysfs attributes production reads from it.
  static void SysfsDevice(const std::string& busId, const std::string& path, const std::string& cls,
                          const std::string& speed = "32.0 GT/s PCIe", const std::string& width = "16") {
    g_pciPaths[busId] = path;
    g_sysfs[path + "/class"] = cls;
    g_sysfs[path + "/vendor"] = "0x1002";
    g_sysfs[path + "/device"] = "0x74a1";
    g_sysfs[path + "/subsystem_vendor"] = "0x1002";
    g_sysfs[path + "/subsystem_device"] = "0x74a2";
    g_sysfs[path + "/max_link_speed"] = speed;
    g_sysfs[path + "/../max_link_speed"] = speed;
    g_sysfs[path + "/max_link_width"] = width;
    g_sysfs[path + "/../max_link_width"] = width;
  }

  ncclXmlNode* PciNode(const char* busId) {
    ncclXmlNode* node = nullptr;
    EXPECT_EQ(ncclSuccess, ncclTopoGetPciNode(xml_.get(), busId, &node));
    return node;
  }
};

TEST_F(XmlSysfsMicrotest, GetXmlFromCpu_FillsAffinityArchAndCpuid) {
  ncclXmlNode* cpu = Add(system_, "cpu", {{"numaid", "3"}});
  g_numaCpumap[3] = "0000ffff";
  g_numaCpuOffset = 64;
  CpuId host = HostCpuId();

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromCpu(cpu, xml_.get()));

  EXPECT_EQ("0000ffff", Attr(cpu, "affinity"));
  EXPECT_EQ("64", Attr(cpu, "affinity_offset"));
  EXPECT_EQ("x86_64", Attr(cpu, "arch"));
  EXPECT_EQ(host.vendor, Attr(cpu, "vendor"));
  EXPECT_EQ(std::to_string(host.family), Attr(cpu, "familyid"));
  EXPECT_EQ(std::to_string(host.model), Attr(cpu, "modelid"));
}

TEST_F(XmlSysfsMicrotest, GetXmlFromCpu_KeepsWhatTheTopologyFileProvides) {
  ncclXmlNode* cpu = Add(system_, "cpu", {{"affinity", "1"}, {"arch", "arm64"}, {"vendor", "V"}, {"familyid", "7"}});
  g_numaCpumap[0] = "ffff";

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromCpu(cpu, xml_.get()));

  EXPECT_EQ(4, cpu->nAttrs);
  EXPECT_EQ("1", Attr(cpu, "affinity"));
  EXPECT_EQ("arm64", Attr(cpu, "arch"));
  EXPECT_EQ("V", Attr(cpu, "vendor"));
}

TEST_F(XmlSysfsMicrotest, GetXmlFromCpu_ZeroOffsetSetsNoOffsetAttribute) {
  ncclXmlNode* cpu = Add(system_, "cpu", {{"numaid", "0"}});

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromCpu(cpu, xml_.get()));

  EXPECT_EQ("", Attr(cpu, "affinity"));
  EXPECT_EQ("<unset>", Attr(cpu, "affinity_offset"));
}

TEST_F(XmlSysfsMicrotest, GetXmlFromCpu_NeedsANumaIdWithoutAffinity) {
  ncclXmlNode* cpu = Add(system_, "cpu");

  EXPECT_EQ(ncclInternalError, ncclTopoGetXmlFromCpu(cpu, xml_.get()));

  EXPECT_TRUE(Logged("could not find CPU numa ID"));
}

TEST_F(XmlSysfsMicrotest, SetAttrFromNvml_SplitsVendorAndDeviceHalves) {
  ncclXmlNode* pci = PciNode("0000:01:00.0");
  ScopedHook pciInfo(g_nvmlGetPciInfo, [](nvmlDevice_t, nvmlPciInfo_t* info) {
    info->pciDeviceId = 0x74a11002;
    info->pciSubSystemId = 0x0c341003;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, ncclTopoSetAttrFromNvml(pci, nullptr, "vendor"));
  EXPECT_EQ(ncclSuccess, ncclTopoSetAttrFromNvml(pci, nullptr, "device"));
  EXPECT_EQ(ncclSuccess, ncclTopoSetAttrFromNvml(pci, nullptr, "subsystem_vendor"));
  EXPECT_EQ(ncclSuccess, ncclTopoSetAttrFromNvml(pci, nullptr, "subsystem_device"));
  EXPECT_EQ(ncclInternalError, ncclTopoSetAttrFromNvml(pci, nullptr, "class"));

  EXPECT_EQ("0x1002", Attr(pci, "vendor"));
  EXPECT_EQ("0x74a1", Attr(pci, "device"));
  EXPECT_EQ("0x1003", Attr(pci, "subsystem_vendor"));
  EXPECT_EQ("0xc34", Attr(pci, "subsystem_device"));
  EXPECT_EQ("<unset>", Attr(pci, "class"));
}

TEST_F(XmlSysfsMicrotest, SetAttrFromNvml_ReturnsTheNvmlError) {
  EXPECT_EQ(ncclSystemError, ncclTopoSetAttrFromNvml(PciNode("0000:01:00.0"), nullptr, "vendor"));
}

TEST_F(XmlSysfsMicrotest, GetPciNode_ReusesTheNodeForABusId) {
  ncclXmlNode* first = PciNode("0000:01:00.0");

  EXPECT_EQ(first, PciNode("0000:01:00.0"));
  EXPECT_NE(first, PciNode("0000:02:00.0"));
  EXPECT_EQ(nullptr, first->parent);
}

// Switches show up as an upstream/downstream port pair, so the walk climbs two path levels at a time.
TEST_F(XmlSysfsMicrotest, GetXmlFromSys_ClimbsSwitchesToTheNumaNodeCpu) {
  const std::string root = "/sys/devices/pci0000:00/0000:00:01.0";
  SysfsDevice("0000:03:00.0", root + "/0000:01:00.0/0000:02:00.0/0000:03:00.0", "0x120000", "32.0 GT/s PCIe", "16");
  g_sysfs[root + "/0000:01:00.0/0000:02:00.0/0000:03:00.0/../max_link_speed"] = "16.0 GT/s PCIe";
  g_sysfs[root + "/0000:01:00.0/0000:02:00.0/0000:03:00.0/../max_link_width"] = "8";
  SysfsDevice("0000:01:00.0", root + "/0000:01:00.0", "0x060400");
  g_sysfs[root + "/0000:01:00.0/numa_node"] = "1";
  g_numaCpumap[1] = "ff00";
  ncclXmlNode* gpu = PciNode("0000:03:00.0");

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(gpu, xml_.get()));

  EXPECT_EQ("0x120000", Attr(gpu, "class"));
  EXPECT_EQ("0x74a2", Attr(gpu, "subsystem_device"));
  EXPECT_EQ("16.0 GT/s PCIe", Attr(gpu, "link_speed"));
  EXPECT_EQ("8", Attr(gpu, "link_width"));
  ncclXmlNode* sw = gpu->parent;
  ASSERT_NE(nullptr, sw);
  EXPECT_EQ("0000:01:00.0", Attr(sw, "busid"));
  EXPECT_EQ("0x060400", Attr(sw, "class"));
  ncclXmlNode* cpu = sw->parent;
  ASSERT_NE(nullptr, cpu);
  EXPECT_STREQ("cpu", cpu->name);
  EXPECT_EQ("1", Attr(cpu, "numaid"));
  EXPECT_EQ("ff00", Attr(cpu, "affinity"));
  EXPECT_NE("<unset>", Attr(cpu, "host_hash"));
  EXPECT_EQ(system_, cpu->parent);
}

TEST_F(XmlSysfsMicrotest, GetXmlFromSys_TakesTheSlowerOfDeviceAndPortLink) {
  const std::string path = "/sys/devices/pci0000:00/0000:00:01.0";
  SysfsDevice("0000:00:01.0", path, "0x020000", "8.0 GT/s PCIe", "4");
  g_sysfs[path + "/numa_node"] = "-1";
  ncclXmlNode* nic = PciNode("0000:00:01.0");

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(nic, xml_.get()));

  EXPECT_EQ("8.0 GT/s PCIe", Attr(nic, "link_speed"));
  EXPECT_EQ("4", Attr(nic, "link_width"));
  EXPECT_EQ("0", Attr(nic->parent, "numaid"));  // the kernel's -1 reads as NUMA node 0
}

TEST_F(XmlSysfsMicrotest, GetXmlFromSys_KeepsTheTopologyFileAttributes) {
  const std::string path = "/sys/devices/pci0000:00/0000:00:01.0";
  SysfsDevice("0000:00:01.0", path, "0x020000");
  ncclXmlNode* pci = PciNode("0000:00:01.0");
  for (const char* key : {"vendor", "device", "subsystem_vendor", "subsystem_device", "link_speed", "link_width"}) {
    ASSERT_EQ(ncclSuccess, xmlSetAttr(pci, key, "file"));
  }

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(pci, xml_.get()));

  EXPECT_EQ("file", Attr(pci, "vendor"));
  EXPECT_EQ("file", Attr(pci, "subsystem_device"));
  EXPECT_EQ("file", Attr(pci, "link_speed"));
  EXPECT_EQ("file", Attr(pci, "link_width"));
}

TEST_F(XmlSysfsMicrotest, GetXmlFromSys_NoSysfsAttachesToTheUnknownCpu) {
  ncclXmlNode* a = PciNode("0000:01:00.0");
  ASSERT_EQ(ncclSuccess, xmlSetAttr(a, "class", "0x020000"));
  ncclXmlNode* b = PciNode("0000:02:00.0");
  ASSERT_EQ(ncclSuccess, xmlSetAttr(b, "class", "0x020000"));

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(b, xml_.get()));
  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(a, xml_.get()));

  EXPECT_EQ("", Attr(a, "link_speed"));
  EXPECT_EQ("", Attr(a, "link_width"));
  EXPECT_EQ("<unset>", Attr(a, "vendor"));
  ncclXmlNode* cpu = a->parent;
  ASSERT_NE(nullptr, cpu);
  EXPECT_EQ(cpu, b->parent);
  EXPECT_EQ("-1", Attr(cpu, "numaid"));
  EXPECT_EQ("x86_64", Attr(cpu, "arch"));
  ASSERT_EQ(2, cpu->nSubs);
  EXPECT_EQ(a, cpu->subs[0]);  // ordered by bus id, not by arrival
  EXPECT_EQ(b, cpu->subs[1]);
}

TEST_F(XmlSysfsMicrotest, GetXmlFromSys_GpuClassReadsPciDataFromNvml) {
  ncclXmlNode* gpu = PciNode("0000:01:00.0");
  ASSERT_EQ(ncclSuccess, xmlSetAttr(gpu, "class", "0x030000"));
  ScopedHook handle(g_nvmlGetHandleByPciBusId, [](const char*, nvmlDevice_t* device) {
    *device = reinterpret_cast<nvmlDevice_t>(0x1);
    return ncclSuccess;
  });
  ScopedHook pciInfo(g_nvmlGetPciInfo, [](nvmlDevice_t, nvmlPciInfo_t* info) {
    info->pciDeviceId = 0x233010de;
    info->pciSubSystemId = 0x16c110de;
    return ncclSuccess;
  });
  ScopedHook gen(g_nvmlGetPcieLinkGeneration, [](nvmlDevice_t, unsigned int* g) {
    *g = 6;
    return ncclSuccess;
  });
  ScopedHook width(g_nvmlGetPcieLinkWidth, [](nvmlDevice_t, unsigned int* w) {
    *w = 8;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(gpu, xml_.get()));

  EXPECT_EQ("0x10de", Attr(gpu, "vendor"));
  EXPECT_EQ("0x2330", Attr(gpu, "device"));
  EXPECT_EQ("0x16c1", Attr(gpu, "subsystem_device"));
  EXPECT_EQ("64.0 GT/s PCIe", Attr(gpu, "link_speed"));
  EXPECT_EQ("8", Attr(gpu, "link_width"));
}

TEST_F(XmlSysfsMicrotest, GetXmlFromSys_NvmlLinkFailuresDefaultToGen4X16) {
  ncclXmlNode* gpu = PciNode("0000:01:00.0");
  ScopedHook handle(g_nvmlGetHandleByPciBusId, [](const char*, nvmlDevice_t*) { return ncclSuccess; });

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(gpu, xml_.get()));

  EXPECT_EQ("16.0 GT/s PCIe", Attr(gpu, "link_speed"));
  EXPECT_EQ("16", Attr(gpu, "link_width"));
  EXPECT_EQ("<unset>", Attr(gpu, "vendor"));  // the PCI info query failed too
}

TEST_F(XmlSysfsMicrotest, GetXmlFromSys_UnknownNvmlLinkGenerationLeavesSpeedUnset) {
  ncclXmlNode* gpu = PciNode("0000:01:00.0");
  ScopedHook handle(g_nvmlGetHandleByPciBusId, [](const char*, nvmlDevice_t*) { return ncclSuccess; });
  ScopedHook gen(g_nvmlGetPcieLinkGeneration, [](nvmlDevice_t, unsigned int* g) {
    *g = 7;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(gpu, xml_.get()));

  EXPECT_EQ("<unset>", Attr(gpu, "link_speed"));
}

TEST_F(XmlSysfsMicrotest, GetXmlFromSys_BroadcomSwitchGetsAPcilinkPerPeer) {
  ncclXmlNode* sw = PciNode("0000:05:00.0");
  ASSERT_EQ(ncclSuccess, xmlSetAttr(sw, "vendor", "0x1000"));
  ASSERT_EQ(ncclSuccess, xmlSetAttr(sw, "class", "0x060400"));
  ncclXmlNode* existing = Add(sw, "pcilink", {{"target", "0000:06:00.0"}});
  g_bcmLinks["0000:05:00.0"] = {"0000:06:00.0", "0000:07:00.0"};

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(sw, xml_.get()));

  ASSERT_EQ(2, sw->nSubs);
  EXPECT_EQ(existing, sw->subs[0]);
  EXPECT_EQ("0000:07:00.0", Attr(sw->subs[1], "target"));
}

TEST_F(XmlSysfsMicrotest, GetXmlFromSys_ExistingParentIsFilledInTurn) {
  ncclXmlNode* cpu = Add(system_, "cpu", {{"numaid", "2"}});
  ncclXmlNode* pci = Add(cpu, "pci", {{"busid", "0000:01:00.0"}, {"class", "0x020000"}});
  ncclXmlNode* orphan = Add(system_, "pci", {{"busid", "0000:02:00.0"}, {"class", "0x020000"}});

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(pci, xml_.get()));
  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromSys(orphan, xml_.get()));

  EXPECT_EQ("x86_64", Attr(cpu, "arch"));
  EXPECT_EQ(system_, orphan->parent);
}

TEST_F(XmlSysfsMicrotest, GetXmlFromSys_FullCpuFails) {
  ncclXmlNode* cpu = Add(system_, "cpu", {{"numaid", "-1"}});
  ncclXmlNode* filler = Add(nullptr, "pci", {{"busid", "ffff:ff:ff.f"}});
  for (int s = 0; s < MAX_SUBS; s++) cpu->subs[s] = filler;
  cpu->nSubs = MAX_SUBS;
  ncclXmlNode* pci = PciNode("0000:01:00.0");
  ASSERT_EQ(ncclSuccess, xmlSetAttr(pci, "class", "0x020000"));

  EXPECT_EQ(ncclInternalError, ncclTopoGetXmlFromSys(pci, xml_.get()));

  EXPECT_TRUE(Logged("XML parser is limited to 512 subnodes"));
}

TEST_F(XmlSysfsMicrotest, GetXmlFromSys_FullNodeFailsToRecordTheLink) {
  ncclXmlNode* pci = PciNode("0000:01:00.0");
  ASSERT_EQ(ncclSuccess, xmlSetAttr(pci, "class", "0x020000"));
  for (int a = pci->nAttrs; a < MAX_ATTR_COUNT; a++) {
    ASSERT_EQ(ncclSuccess, xmlSetAttrInt(pci, std::to_string(a).c_str(), a));
  }

  EXPECT_EQ(ncclInternalError, ncclTopoGetXmlFromSys(pci, xml_.get()));
}

// --- Autodetection: GPUs

class XmlGpuMicrotest : public XmlSysfsMicrotest {
 protected:
  ncclXmlNode* pci_ = nullptr;
  std::vector<std::string> smiBusIds_ = {"0000:10:00.0", "0000:20:00.0", "0000:30:00.0"};
  std::map<std::pair<int, int>, std::pair<amdsmi_link_type_t, int>> links_;  // (src, dst) -> (type, hops)

  void SetUp() override {
    XmlSysfsMicrotest::SetUp();
    ncclXmlNode* cpu = Add(system_, "cpu", {{"numaid", "0"}});
    pci_ = Add(cpu, "pci", {{"busid", "0000:10:00.0"}});
    g_amdSmiGetNumDevice = [this](uint32_t* n) {
      *n = smiBusIds_.size();
      return ncclSuccess;
    };
    g_amdSmiGetDevicePciBusIdString = [this](uint32_t i, char* busId, size_t len) {
      snprintf(busId, len, "%s", smiBusIds_[i].c_str());
      return ncclSuccess;
    };
    g_amdSmiGetLinkInfo = [this](int src, int dst, amdsmi_link_type_t* type, int* hops, int* count) {
      auto it = links_.find({src, dst});
      if (it == links_.end()) return ncclInternalError;
      *type = it->second.first;
      *hops = it->second.second;
      *count = 2;
      return ncclSuccess;
    };
  }

  ncclXmlNode* GpuNode(uint32_t dev = 0, uint32_t smi = 0, ncclResult_t expect = ncclSuccess) {
    ncclXmlNode* gpu = nullptr;
    EXPECT_EQ(expect, ncclTopoGetXmlFromGpu(pci_, dev, smi, xml_.get(), &gpu));
    return gpu;
  }

  static std::vector<ncclXmlNode*> Links(ncclXmlNode* gpu) {
    std::vector<ncclXmlNode*> links;
    for (int s = 0; s < gpu->nSubs; s++) {
      if (strcmp(gpu->subs[s]->name, "xgmi") == 0) links.push_back(gpu->subs[s]);
    }
    return links;
  }
};

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_FillsSmGcnAndArchFromHip) {
  hipDeviceArch_t arch{};
  arch.hasGlobalInt32Atomics = 1;
  arch.hasDoubles = 1;
  ScopedHook props(g_hipGetDeviceProperties, [arch](hipDeviceProp_t* prop, int dev) {
    *prop = hipDeviceProp_t{};
    prop->multiProcessorCount = 80 + dev;
    prop->arch = arch;
    snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "gfx950:sramecc+:xnack-");
    return hipSuccess;
  });
  rcclHipDeviceArch_t expected;
  expected.arch = arch;

  ncclXmlNode* gpu = GpuNode(2, 7);

  ASSERT_NE(nullptr, gpu);
  EXPECT_EQ(pci_, gpu->parent);
  EXPECT_EQ("7", Attr(gpu, "dev"));  // the SMI index, the same in every process
  EXPECT_EQ("82", Attr(gpu, "sm"));  // properties come from the HIP ordinal
  EXPECT_EQ("gfx950", Attr(gpu, "gcn"));
  EXPECT_EQ(std::to_string(expected.value), Attr(gpu, "arch"));
  EXPECT_EQ(3, props.calls);
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_TopologyFileValuesSkipHip) {
  ncclXmlNode* file = Add(pci_, "gpu", {{"dev", "0"}, {"sm", "1"}, {"gcn", "942"}, {"arch", "5"}});
  Add(file, "xgmi", {{"target", "0000:20:00.0"}, {"count", "1"}, {"tclass", "0x120000"}});
  ScopedHook props(g_hipGetDeviceProperties, [](hipDeviceProp_t*, int) { return hipErrorInvalidDevice; });
  ScopedHook numDevice(g_amdSmiGetNumDevice, [](uint32_t*) { return ncclInternalError; });

  ncclXmlNode* gpu = GpuNode();

  EXPECT_EQ(file, gpu);
  EXPECT_EQ("gfx942", Attr(gpu, "gcn"));  // legacy numeric gcn is renamed
  EXPECT_EQ(0, props.calls);
  EXPECT_EQ(0, numDevice.calls);
  EXPECT_EQ(1u, Links(gpu).size());
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_DevMinusOneHasNoGpu) {
  EXPECT_EQ(nullptr, GpuNode(0, UINT32_MAX));
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_HipPropertiesFailureIsReturned) {
  ScopedHook props(g_hipGetDeviceProperties, [](hipDeviceProp_t*, int) { return hipErrorInvalidDevice; });

  GpuNode(0, 0, ncclUnhandledCudaError);
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_OneHopXgmiPeersBecomeLinks) {
  links_[{0, 1}] = {AMDSMI_LINK_TYPE_XGMI, 1};
  links_[{0, 2}] = {AMDSMI_LINK_TYPE_XGMI, 2};
  smiBusIds_.push_back("0000:AB:00.0");
  links_[{0, 3}] = {AMDSMI_LINK_TYPE_XGMI, 1};
  smiBusIds_.push_back("0000:40:00.0");
  links_[{0, 4}] = {AMDSMI_LINK_TYPE_PCIE, 1};

  ncclXmlNode* gpu = GpuNode();

  std::vector<ncclXmlNode*> links = Links(gpu);
  ASSERT_EQ(2u, links.size());
  EXPECT_EQ("0000:20:00.0", Attr(links[0], "target"));
  EXPECT_EQ("2", Attr(links[0], "count"));
  EXPECT_EQ("0", Attr(links[0], "fabric_supported"));
  EXPECT_EQ(PCI_ACCELERATOR_CLASS, Attr(links[0], "tclass"));
  EXPECT_EQ("0000:ab:00.0", Attr(links[1], "target"));
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_SharedFabricCliqueMakesUaloeLinks) {
  smiBusIds_[2] = smiBusIds_[1];  // a second report of the same peer must not restamp it
  ScopedHook fabric(g_amdSmiGetFabricDeviceInfo, [](uint32_t i, amdsmiFabricDeviceInfo* info) {
    *info = amdsmiFabricDeviceInfo{};
    info->fabricSupported = true;
    info->cliqueId = 4;
    for (int b = 0; b < UALOE_GPU_FABRIC_UUID_LEN; b++) info->clusterUuid[b] = b + 1;
    info->acceleratorId = 10 + i;
    info->bandwidth = 400;
    info->latency = 3;
    info->ppodSize = 16;
    info->vpodSize = 8;
    return ncclSuccess;
  });

  ncclXmlNode* gpu = GpuNode();

  std::vector<ncclXmlNode*> links = Links(gpu);
  ASSERT_EQ(1u, links.size());
  ncclXmlNode* link = links[0];
  EXPECT_EQ("1", Attr(link, "count"));
  EXPECT_EQ("1", Attr(link, "fabric_supported"));
  EXPECT_EQ("11", Attr(link, "accelerator_id"));
  EXPECT_EQ("400", Attr(link, "bandwidth"));
  EXPECT_EQ("3", Attr(link, "latency"));
  EXPECT_EQ("0x807060504030201", Attr(link, "uuidHigh"));
  EXPECT_EQ("0x100f0e0d0c0b0a09", Attr(link, "uuidLow"));
  EXPECT_EQ("16", Attr(link, "ppod_size"));
  EXPECT_EQ("4", Attr(link, "vpod_id"));
  EXPECT_EQ("8", Attr(link, "vpod_size"));
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_FabricMismatchFallsBackToXgmiLinks) {
  links_[{0, 1}] = {AMDSMI_LINK_TYPE_XGMI, 1};
  links_[{0, 2}] = {AMDSMI_LINK_TYPE_XGMI, 1};
  smiBusIds_.push_back("0000:40:00.0");
  links_[{0, 3}] = {AMDSMI_LINK_TYPE_XGMI, 1};
  ScopedHook fabric(g_amdSmiGetFabricDeviceInfo, [](uint32_t i, amdsmiFabricDeviceInfo* info) {
    *info = amdsmiFabricDeviceInfo{};
    info->fabricSupported = i != 1;   // 1: peer has no fabric
    info->cliqueId = i == 2 ? 9 : 4;  // 2: other clique
    info->clusterUuid[0] = i == 3;    // 3: other cluster
    return ncclSuccess;
  });

  std::vector<ncclXmlNode*> links = Links(GpuNode());

  ASSERT_EQ(3u, links.size());
  for (ncclXmlNode* link : links) EXPECT_EQ("0", Attr(link, "fabric_supported"));
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_FabricQueryFailureFallsBackToXgmiLinks) {
  links_[{0, 1}] = {AMDSMI_LINK_TYPE_XGMI, 1};
  ScopedHook fabric(g_amdSmiGetFabricDeviceInfo, [](uint32_t i, amdsmiFabricDeviceInfo* info) {
    info->fabricSupported = true;
    return i == 0 ? ncclSuccess : ncclInternalError;
  });

  EXPECT_EQ(1u, Links(GpuNode()).size());
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_HipPartitionSiblingsGetOneHopXgmi) {
  const std::vector<std::string> hipBusIds = {"0000:10:00.0", "0000:10:00.1", "0000:20:00.0", "", "0000:10:00.3"};
  ScopedHook count(g_hipGetDeviceCount, [&](int* n) {
    *n = hipBusIds.size();
    return hipSuccess;
  });
  ScopedHook busId(g_hipDeviceGetPCIBusId, [&](char* busId, int len, int dev) {
    if (hipBusIds[dev].empty()) return hipErrorInvalidDevice;
    snprintf(busId, len, "%s", hipBusIds[dev].c_str());
    return hipSuccess;
  });
  ncclXmlNode* gpu = Add(pci_, "gpu", {{"dev", "0"}, {"sm", "1"}, {"gcn", "gfx942"}, {"arch", "0"}});
  Add(gpu, "xgmi", {{"target", "0000:10:00.3"}, {"count", "1"}, {"tclass", "0x120000"}, {"fabric_supported", "1"}});

  ASSERT_EQ(gpu, GpuNode());

  std::vector<ncclXmlNode*> links = Links(gpu);
  ASSERT_EQ(2u, links.size());
  EXPECT_EQ("1", Attr(links[0], "fabric_supported"));  // already known, left alone
  EXPECT_EQ("0000:10:00.1", Attr(links[1], "target"));
  EXPECT_EQ("1", Attr(links[1], "count"));
  EXPECT_EQ("0", Attr(links[1], "fabric_supported"));
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_PartitionMatchesOnTheDeviceNotTheFunction) {
  ScopedHook count(g_hipGetDeviceCount, [](int* n) {
    *n = 2;
    return hipSuccess;
  });
  ScopedHook busId(g_hipDeviceGetPCIBusId, [](char* busId, int len, int dev) {
    snprintf(busId, len, "0000:10:00.%d", dev);
    return hipSuccess;
  });
  pci_ = Add(pci_->parent, "pci", {{"busid", "0000:10:00.1"}});

  std::vector<ncclXmlNode*> links = Links(GpuNode(1));

  ASSERT_EQ(1u, links.size());
  EXPECT_EQ("0000:10:00.0", Attr(links[0], "target"));
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_PartitionScanNeedsABusIdAndADeviceCount) {
  ScopedHook count(g_hipGetDeviceCount, [](int*) { return hipErrorNoDevice; });

  EXPECT_TRUE(Links(GpuNode()).empty());
  EXPECT_EQ(1, count.calls);

  ncclXmlNode* bare = Add(pci_->parent, "pci");
  ncclXmlNode* gpu = nullptr;
  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromGpu(bare, 0, 0, xml_.get(), &gpu));
  EXPECT_EQ(1, count.calls);
}

TEST_F(XmlGpuMicrotest, GetXmlFromGpu_MissingTargetClassComesFromSysfs) {
  g_pciPaths["0000:20:00.0"] = "/sys/x/0000:20:00.0";
  g_sysfs["/sys/x/0000:20:00.0/class"] = "0x038000";
  g_pciPaths["0000:30:00.0"] = "/sys/x/0000:30:00.0";
  g_sysfs["/sys/x/0000:30:00.0/class"] = "0x020000";
  ncclXmlNode* gpu = Add(pci_, "gpu", {{"dev", "0"}, {"sm", "1"}, {"gcn", "gfx942"}, {"arch", "0"}});
  for (const char* target : {"fffffff:ffff:ff", "0000:20:00.0", "0000:30:00.0", "0000:40:00.0"}) {
    Add(gpu, "xgmi", {{"target", target}, {"count", "1"}});
  }
  Add(gpu, "other");

  ASSERT_EQ(gpu, GpuNode());

  std::vector<ncclXmlNode*> links = Links(gpu);
  ASSERT_EQ(4u, links.size());
  EXPECT_EQ(PCI_NVSWITCH_CLASS, Attr(links[0], "tclass"));
  EXPECT_EQ("0x038000", Attr(links[1], "tclass"));
  EXPECT_EQ(PCI_ACCELERATOR_CLASS, Attr(links[2], "tclass"));  // sysfs says NIC: a HIP-only BDF
  EXPECT_EQ(PCI_ACCELERATOR_CLASS, Attr(links[3], "tclass"));  // not in sysfs at all
  EXPECT_TRUE(Logged("sysfs class 0x020000 is not a GPU"));
  EXPECT_TRUE(Logged("No sysfs PCI class for XGMI target 0000:40:00.0"));
}

TEST_F(XmlGpuMicrotest, FillGpu_BuildsThePciChainWithTheSmiIndexAsDev) {
  ncclXmlNode* top = nullptr;
  ScopedHook ordinal(g_hipDeviceGetByPCIBusId, [](int* dev, const char*) {
    *dev = 5;
    return hipSuccess;
  });
  std::vector<int> propDevs;
  ScopedHook props(g_hipGetDeviceProperties, [&propDevs](hipDeviceProp_t* prop, int dev) {
    *prop = hipDeviceProp_t{};
    snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "gfx942");
    propDevs.push_back(dev);
    return hipSuccess;
  });
  ScopedHook index(g_amdSmiGetDeviceIndexByPciBusId, [](const char* busId, uint32_t* i) {
    EXPECT_STREQ("0000:50:00.0", busId);
    *i = 1;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, ncclTopoFillGpu(xml_.get(), "0000:50:00.0", &top));

  ASSERT_NE(nullptr, top);
  EXPECT_EQ("1", Attr(top, "dev"));
  ASSERT_FALSE(propDevs.empty());
  for (int dev : propDevs) EXPECT_EQ(5, dev);
  EXPECT_EQ("0x03", Attr(top->parent, "class"));
  EXPECT_EQ("0000:50:00.0", Attr(top->parent, "busid"));
  EXPECT_EQ(1, g_amdSmiInitCalls);
}

// The static amdsmiInit guard is never set, so every call runs amd_smi_init again; a fix flips this pin.
TEST_F(XmlGpuMicrotest, FillGpu_KnownBug_AmdSmiInitRunsOnEveryCall) {
  ncclXmlNode* top = nullptr;
  ScopedHook ordinal(g_hipDeviceGetByPCIBusId, [](int* dev, const char*) {
    *dev = 5;
    return hipSuccess;
  });

  ASSERT_EQ(ncclSuccess, ncclTopoFillGpu(xml_.get(), "0000:50:00.0", &top));
  ASSERT_EQ(ncclSuccess, ncclTopoFillGpu(xml_.get(), "0000:50:00.0", &top));

  EXPECT_EQ(2, g_amdSmiInitCalls);
}

TEST_F(XmlGpuMicrotest, FillGpu_HipLookupFailureIsAnError) {
  ncclXmlNode* gpu = nullptr;

  EXPECT_EQ(ncclUnhandledCudaError, ncclTopoFillGpu(xml_.get(), "0000:50:00.0", &gpu));

  EXPECT_EQ(nullptr, gpu);
}

TEST_F(XmlGpuMicrotest, FillGpu_HipLogicalFunctionAttachesToFunctionZero) {
  ncclXmlNode* gpu = nullptr;
  ScopedHook ordinal(g_hipDeviceGetByPCIBusId, [](int* dev, const char*) {
    *dev = 0;
    return hipSuccess;
  });
  g_pciPaths["0000:50:00.2"] = "/sys/x/0000:50:00.2";
  g_sysfs["/sys/x/0000:50:00.2/class"] = "0x020000";

  ASSERT_EQ(ncclSuccess, ncclTopoFillGpu(xml_.get(), "0000:50:00.1", &gpu));
  ncclXmlNode* first = gpu->parent;
  ASSERT_EQ(ncclSuccess, ncclTopoFillGpu(xml_.get(), "0000:50:00.2", &gpu));

  EXPECT_EQ("0000:50:00.0", Attr(first, "busid"));
  EXPECT_EQ(first, gpu->parent);
  EXPECT_TRUE(Logged("HIP logical GPU at 0000:50:00.1 is not a physical GPU PCI function (class missing)"));
  EXPECT_TRUE(Logged("(class 0x020000); attaching under 0000:50:00.0"));
}

TEST_F(XmlGpuMicrotest, FillGpu_GpuClassFunctionKeepsItsOwnBusId) {
  ncclXmlNode* gpu = nullptr;
  ScopedHook ordinal(g_hipDeviceGetByPCIBusId, [](int* dev, const char*) {
    *dev = 0;
    return hipSuccess;
  });
  g_pciPaths["0000:50:00.1"] = "/sys/x/0000:50:00.1";
  g_sysfs["/sys/x/0000:50:00.1/class"] = "0x120000";
  g_pciPaths["0000:50:00.8"] = "/sys/x/0000:50:00.8";

  ASSERT_EQ(ncclSuccess, ncclTopoFillGpu(xml_.get(), "0000:50:00.1", &gpu));
  EXPECT_EQ("0000:50:00.1", Attr(gpu->parent, "busid"));
  ASSERT_EQ(ncclSuccess, ncclTopoFillGpu(xml_.get(), "0000:50:00.8", &gpu));
  EXPECT_EQ("0000:50:00.8", Attr(gpu->parent, "busid"));  // past the partition range
}

TEST_F(XmlGpuMicrotest, FillGpu_SmiFailuresAreReturned) {
  ncclXmlNode* gpu = nullptr;
  g_amdSmiInitResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ncclTopoFillGpu(xml_.get(), "0000:50:00.0", &gpu));

  g_amdSmiInitResult = ncclSuccess;
  ScopedHook index(g_amdSmiGetDeviceIndexByPciBusId, [](const char*, uint32_t*) { return ncclInvalidArgument; });
  EXPECT_EQ(ncclInvalidArgument, ncclTopoFillGpu(xml_.get(), "0000:50:00.0", &gpu));
}

// --- Autodetection: NICs

TEST_F(XmlSysfsMicrotest, FillNet_PciNicHangsUnderItsPciDevice) {
  const std::string path = "/sys/devices/pci0000:00/0000:00:02.0";
  SysfsDevice("0000:00:02.0", path, "");
  g_realpaths[path + "/subsystem"] = "/sys/bus/pci";
  ncclXmlNode* net = nullptr;
  ncclXmlNode* again = nullptr;

  ASSERT_EQ(ncclSuccess, ncclTopoFillNet(xml_.get(), "net", path.c_str(), "mlx5_0", &net));
  ASSERT_EQ(ncclSuccess, ncclTopoFillNet(xml_.get(), "net", path.c_str(), "mlx5_0", &again));

  EXPECT_EQ(net, again);
  EXPECT_EQ("mlx5_0", Attr(net, "name"));
  ncclXmlNode* nic = net->parent;
  EXPECT_STREQ("nic", nic->name);
  EXPECT_EQ("0000:00:02.0", Attr(nic->parent, "busid"));
  EXPECT_EQ("0x02", Attr(nic->parent, "class"));
  EXPECT_STREQ("cpu", nic->parent->parent->name);
}

TEST_F(XmlSysfsMicrotest, FillNet_NonPciDeviceGoesToTheFirstCpu) {
  ncclXmlNode* cpu = Add(system_, "cpu", {{"numaid", "1"}});
  g_realpaths["/sys/devices/virtual/net/ib0/subsystem"] = "/sys/class/net";
  ncclXmlNode* a = nullptr;
  ncclXmlNode* b = nullptr;

  ASSERT_EQ(ncclSuccess, ncclTopoFillNet(xml_.get(), "net", "/sys/devices/virtual/net/ib0", "ib0", &a));
  ASSERT_EQ(ncclSuccess, ncclTopoFillNet(xml_.get(), "gin", "/sys/devices/usb/x", "x", &b));

  EXPECT_EQ(cpu, a->parent->parent);
  EXPECT_EQ(a->parent, b->parent);  // one nic per parent
  EXPECT_STREQ("gin", b->name);
  EXPECT_TRUE(Logged("is not a PCI device (net)"));
}

TEST_F(XmlSysfsMicrotest, FillNet_WithoutACpuCreatesNumaZero) {
  ncclXmlNode* net = nullptr;

  ASSERT_EQ(ncclSuccess, ncclTopoFillNet(xml_.get(), "net", nullptr, "virt0", &net));

  ncclXmlNode* cpu = net->parent->parent;
  EXPECT_EQ("0", Attr(cpu, "numaid"));
  EXPECT_NE("<unset>", Attr(cpu, "host_hash"));
  EXPECT_EQ(system_, cpu->parent);
}

TEST_F(XmlSysfsMicrotest, FillNet_ForcedParentWins) {
  ncclXmlNode* parent = Add(system_, "pci", {{"busid", "0000:09:00.0"}});
  ncclXmlNode* net = nullptr;

  ASSERT_EQ(ncclSuccess, ncclTopoFillNet(xml_.get(), "net", "/unused", "v0", &net, parent));

  EXPECT_EQ(parent, net->parent->parent);
}

TEST(XmlFillNetMicrotest, FillNet_NoSystemNoParentFails) {
  ResetGraphFakes();
  XmlPtr xml = NewXml();
  ncclXmlNode* net = nullptr;

  EXPECT_EQ(ncclInternalError, ncclTopoFillNet(xml.get(), "net", nullptr, "v0", &net));

  EXPECT_TRUE(Logged("parent is NULL even after fallback for v0"));
  ResetGraphFakes();
}

}  // namespace
