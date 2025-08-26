/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hipAPI.h"

FileOccurrence::FileOccurrence(std::string file_name, int line_number)
    : file_name{file_name}, line_number{line_number} {}

TestCaseOccurrence::TestCaseOccurrence(std::string test_case_name, std::string file_name,
                                       int line_number)
    : FileOccurrence{file_name, line_number}, test_case_name{test_case_name} {}

bool operator==(const TestCaseOccurrence& l_test, const TestCaseOccurrence& r_test) {
  return l_test.test_case_name == r_test.test_case_name;
}

bool operator<(const TestCaseOccurrence& l_test, const TestCaseOccurrence& r_test) {
  return l_test.test_case_name < r_test.test_case_name;
}

bool operator==(const HipAPI& l_hip_api, const HipAPI& r_hip_api) {
  return l_hip_api.api_name == r_hip_api.api_name;
}

bool operator<(const HipAPI& l_hip_api, const HipAPI& r_hip_api) {
  return l_hip_api.api_name < r_hip_api.api_name;
}

HipAPI::HipAPI(std::string api_name, bool deprecated_flag, std::string api_group_name,
               std::string file_restriction)
    : api_name{api_name},
      deprecated{deprecated_flag},
      api_group_name{api_group_name},
      file_restriction{file_restriction} {
  test_cases.clear();
}

std::string HipAPI::getName() const { return api_name; }

std::string HipAPI::getGroupName() const { return api_group_name; }

int HipAPI::getNumberOfCalls() const { return file_occurrences.size(); }

std::vector<TestCaseOccurrence> HipAPI::getTestCases() const { return test_cases; }

void HipAPI::addFileOccurrence(FileOccurrence file_occurrence) {
  file_occurrences.push_back(file_occurrence);
}

void HipAPI::addTestCase(TestCaseOccurrence test_case) {
  if (std::find(test_cases.begin(), test_cases.end(), test_case) == test_cases.end()) {
    test_cases.push_back(test_case);
  }
}

bool HipAPI::isDeprecated() const { return deprecated; }

std::string HipAPI::getBasicStatsXML() const {
  std::stringstream xml_node;
  xml_node << "\t\t<HIP-API>\n";

  if (!deprecated) {
    xml_node << "\t\t\t<NAME>" << api_name << "</NAME>\n";
  } else {
    xml_node << "\t\t\t<NAME>"
             << "[DEPRECATED] " << api_name << "</NAME>\n";
  }

  if (!file_occurrences.empty()) {
    xml_node << "\t\t\t<NUMBER-OF-API-CALLS>" << file_occurrences.size()
             << "</NUMBER-OF-API-CALLS>\n";
    xml_node << "\t\t\t<FILE-OCCURRENCES>\n";
    for (auto const& file_occurrence : file_occurrences) {
      xml_node << "\t\t\t\t<FILE-OCCURRENCE>" << file_occurrence.file_name << ":"
               << file_occurrence.line_number << "</FILE-OCCURRENCE>\n";
    }
    xml_node << "\t\t\t</FILE-OCCURRENCES>\n";
  }

  xml_node << "\t\t</HIP-API>\n";
  return xml_node.str();
}

std::string HipAPI::createHTMLReport() const {
  std::stringstream html_report;
  std::string one_tab{"\n\t"};
  std::string two_tabs{"\n\t\t"};
  std::string three_tabs{"\n\t\t\t"};
  std::string four_tabs{"\n\t\t\t\t"};
  std::string five_tabs{"\n\t\t\t\t\t"};
  std::string six_tabs{"\n\t\t\t\t\t\t"};

  html_report << "<html lang=\"en\">";
  html_report << "<head>" << one_tab
              << "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\">";
  html_report << one_tab << "<title>" << api_name << " Coverage report</title>" << one_tab
              << "<link rel=\"stylesheet\" type=\"text/css\" href=\"../resources/coverage.css\">"
              << one_tab << "</head>";
  html_report << one_tab << "<body>" << one_tab
              << "<table width=\"100%\" border=0 cellspacing=0 cellpadding=0>";
  html_report << two_tabs << "<tr><td class=\"title\">" << api_name << " Coverage report</td></tr>";
  html_report << two_tabs
              << "<tr><td class=\"ruler\"><img src=\"../resources/glass.png\" width=3 "
                 "height=3></td></tr>\n";
  html_report << two_tabs << "<tr>" << three_tabs << "<td width=\"100%\">" << four_tabs
              << "<table cellpading=1 border=0 width=\"100%\"";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td width=\"70%\"</td>";
  html_report << six_tabs << "<td width=\"10%\"></td>";
  html_report << six_tabs << "<td width=\"10%\"></td>";
  html_report << six_tabs << "<td width=\"10%\"></td>";
  html_report << five_tabs << "</tr>";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td class=\"headerItem\">Group:</td>";
  html_report << six_tabs << "<td class=\"headerValue\" colspan=2>" << api_group_name << "</td>";
  html_report << six_tabs << "<td></td>";
  html_report << five_tabs << "</tr>";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td class=\"headerItem\">Calls within test source files:</td>";
  html_report << six_tabs << "<td class=\"headerCovTableEntry\">" << file_occurrences.size()
              << "</td>";
  html_report << six_tabs << "<td></td>";
  html_report << six_tabs << "<td></td>";
  html_report << five_tabs << "</tr>";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td class=\"headerItem\">Number of test cases:</td>";
  html_report << six_tabs << "<td class=\"headerCovTableEntry\">" << test_cases.size() << "</td>";
  html_report << six_tabs << "<td></td>";
  html_report << six_tabs << "<td></td>";
  html_report << five_tabs << "</tr>";

  html_report << five_tabs
              << "<tr><td><img src=\"../resources/glass.png\" width=3 height=3></td></tr>";
  html_report << four_tabs << "</table>";
  html_report << three_tabs << "</td>";
  html_report << two_tabs << "</tr>";

  html_report << two_tabs
              << "<tr><td class=\"ruler\"><img src=\"../resources/glass.png\" width=3 "
                 "height=3></td></tr>\n";
  html_report << one_tab << "</table>";

  html_report << one_tab << "<center>";
  // Add info about test cases
  html_report << one_tab << "<table width=\"70%\" cellpadding=1 cellspacing=1 border=0>";
  if (!test_cases.empty()) {
    html_report << two_tabs << "<tr>";
    html_report << three_tabs << "<td width=\"20%\"><br></td>";
    html_report << three_tabs << "<td width=\"60%\"><br></td>";
    html_report << three_tabs << "<td width=\"20%\"><br></td>";
    html_report << two_tabs << "</tr>";

    html_report << two_tabs << "<tr>";
    html_report << three_tabs << "<td class=\"tableHead\">Test case ID</td>";
    html_report << three_tabs << "<td class=\"tableHead\">Test case occurrence in file</td>";
    html_report << three_tabs << "<td class=\"tableHead\">Line number</td>";
    html_report << two_tabs << "</tr>";

    for (auto const& test_case : test_cases) {
      html_report << two_tabs << "<tr>";
      html_report << three_tabs << "<td class=\"coverFile\">" << test_case.test_case_name
                  << "</td>";
      html_report << three_tabs << "<td class=\"coverFile\">" << test_case.file_name << "</td>";
      html_report << three_tabs << "<td class=\"headerCovTableEntry\">" << test_case.line_number
                  << "</td>";
      html_report << two_tabs << "</tr>";
    }
  } else {
    html_report << two_tabs << "<tr>";
    html_report << three_tabs
                << "<td class=\"headerItem\" style=\"text-align:center\"><br>There are no test "
                   "cases detected within doxygen comments.</td>";
    html_report << two_tabs << "</tr>";
  }
  html_report << one_tab << "</table>";

  html_report << one_tab << "<br>";
  html_report << one_tab << "<table width=\"100%\" border=0 cellspacing=0 cellpadding=0>";
  html_report
      << two_tabs
      << "<tr><td class=\"ruler\"><img src=\"../resources/glass.png\" width=3 height=3></td></tr>";
  html_report << one_tab << "</table>";

  // Add info about API occurrences in the test files.
  html_report << one_tab << "<table width=\"50%\" cellpadding=1 cellspacing=1 border=0>";
  if (!file_occurrences.empty()) {
    html_report << two_tabs << "<tr>";
    html_report << three_tabs << "<td width=\"80%\"><br></td>";
    html_report << three_tabs << "<td width=\"20%\"><br></td>";
    html_report << two_tabs << "</tr>";

    html_report << two_tabs << "<tr>";
    html_report << three_tabs << "<td class=\"tableHead\">API occurrence in file</td>";
    html_report << three_tabs << "<td class=\"tableHead\">Line number</td>";
    html_report << two_tabs << "</tr>";

    for (auto const& file_occurrence : file_occurrences) {
      html_report << two_tabs << "<tr>";
      html_report << three_tabs << "<td class=\"coverFile\">" << file_occurrence.file_name
                  << "</td>";
      html_report << three_tabs << "<td class=\"headerCovTableEntry\">"
                  << file_occurrence.line_number << "</td>";
      html_report << two_tabs << "</tr>";
    }
  } else {
    html_report << two_tabs << "<tr>";
    html_report << three_tabs
                << "<td class=\"headerItem\" style=\"text-align:center\"><br>There are no "
                   "occurrences within test source files.</td>";
    html_report << two_tabs << "</tr>";
  }
  html_report << one_tab << "</table>";
  html_report << one_tab << "</center>";

  html_report << one_tab << "<br>";
  html_report << one_tab << "<table width=\"100%\" border=0 cellspacing=0 cellpadding=0>";
  html_report
      << two_tabs
      << "<tr><td class=\"ruler\"><img src=\"../resources/glass.png\" width=3 height=3></td></tr>";

  time_t now{time(nullptr)};
  std::string date{asctime(gmtime(&now))};
  html_report << two_tabs << "<tr><td class=\"versionInfo\">Generated: " << date;
  html_report << two_tabs << " UTC</td></tr>";
  html_report << one_tab << "</table>";
  html_report << one_tab << "<br>";
  html_report << "\n</body>\n</html>";

  return html_report.str();
}

std::string HipAPI::getFileRestriction() const { return file_restriction; }
