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

#include "hipAPIGroup.h"

bool operator==(const HipAPIGroup& l_hip_api_group, const HipAPIGroup& r_hip_api_group) {
  return l_hip_api_group.group_name == r_hip_api_group.group_name;
}

HipAPIGroup::HipAPIGroup(std::string group_name, std::vector<HipAPI>& hip_apis)
    : group_name{group_name},
      number_of_api_calls{0},
      percentage_of_called_apis{0.f},
      total_number_of_apis{0},
      number_of_test_cases{0} {
  std::vector<TestCaseOccurrence> test_cases;
  for (auto const& hip_api : hip_apis) {
    if (hip_api.getGroupName() != group_name) {
      continue;
    }

    if (hip_api.isDeprecated()) {
      deprecated_apis.push_back(hip_api);
    } else {
      if (hip_api.getNumberOfCalls()) {
        called_apis.push_back(hip_api);
      } else {
        not_called_apis.push_back(hip_api);
      }
    }

    number_of_api_calls += hip_api.getNumberOfCalls();

    std::vector<TestCaseOccurrence> api_test_cases = hip_api.getTestCases();
    test_cases.insert(test_cases.end(), api_test_cases.begin(), api_test_cases.end());
  }

  std::sort(test_cases.begin(), test_cases.end());
  auto last = std::unique(test_cases.begin(), test_cases.end());
  test_cases.erase(last, test_cases.end());
  number_of_test_cases = test_cases.size();

  total_number_of_apis = called_apis.size() + not_called_apis.size() + deprecated_apis.size();
  if (not_called_apis.empty()) {
    percentage_of_called_apis = 100.f;
  } else {
    percentage_of_called_apis =
        100.f * called_apis.size() / (total_number_of_apis - deprecated_apis.size());
  }
}

std::string HipAPIGroup::getName() const { return group_name; }

int HipAPIGroup::getTotalNumberOfAPIs() const { return total_number_of_apis; }

int HipAPIGroup::getTotalNumberOfCalls() const { return number_of_api_calls; }

int HipAPIGroup::getTotalNumberOfTestCases() const { return number_of_test_cases; }

int HipAPIGroup::getNumberOfCalledAPIs() const { return called_apis.size(); }

int HipAPIGroup::getNumberOfNotCalledAPIs() const { return not_called_apis.size(); }

int HipAPIGroup::getNumberOfDeprecatedAPIs() const { return deprecated_apis.size(); }

float HipAPIGroup::getPercentageOfCalledAPIs() const { return percentage_of_called_apis; }

bool HipAPIGroup::isDeprecated() const {
  return (deprecated_apis.size() == total_number_of_apis) ? true : false;
}

std::string HipAPIGroup::getBasicStatsXML() const {
  std::stringstream xml_node;

  std::string tag_name;
  std::transform(group_name.begin(), group_name.end(), std::back_inserter(tag_name), ::toupper);
  std::replace(tag_name.begin(), tag_name.end(), ' ', '-');

  xml_node << "\n<" << tag_name << ">";

  xml_node << "\n\t<COVERAGE-RESULTS>";
  xml_node << "\n\t\t<TOTAL-NUMBER-OF-APIs>" << total_number_of_apis << "</TOTAL-NUMBER-OF-APIs>";
  xml_node << "\n\t\t<TOTAL-NUMBER-OF-API-CALLS>" << number_of_api_calls
           << "</TOTAL-NUMBER-OF-API-CALLS>";
  xml_node << "\n\t\t<CALLED-APIs>" << called_apis.size() << "</CALLED-APIs>";
  xml_node << "\n\t\t<NOT-CALLED-APIs>" << not_called_apis.size() << "</NOT-CALLED-APIs>";
  xml_node << "\n\t\t<DEPRECATED-APIs>" << deprecated_apis.size() << "</DEPRECATED-APIs>";
  xml_node << "\n\t\t<COVERAGE-PERCENTAGE>" << percentage_of_called_apis
           << "%</COVERAGE-PERCENTAGE>";
  xml_node << "\n\t</COVERAGE-RESULTS>";

  if (!called_apis.empty()) {
    xml_node << "\n\t<LIST-OF-CALLED-APIs>\n";
    for (auto const& hip_api : called_apis) {
      xml_node << hip_api.getBasicStatsXML();
    }
    xml_node << "\t</LIST-OF-CALLED-APIs>";
  }

  if (!not_called_apis.empty()) {
    xml_node << "\n\t<LIST-OF-NOT-CALLED-APIs>\n";
    for (auto const& hip_api : not_called_apis) {
      xml_node << hip_api.getBasicStatsXML();
    }
    xml_node << "\t</LIST-OF-NOT-CALLED-APIs>";
  }

  if (!deprecated_apis.empty()) {
    xml_node << "\n\t<DEPRECATED-APIs>\n";
    for (auto const& hip_api : deprecated_apis) {
      xml_node << hip_api.getBasicStatsXML();
    }
    xml_node << "\t</DEPRECATED-APIs>";
  }

  xml_node << "\n</" << tag_name << ">";
  return xml_node.str();
}

std::string HipAPIGroup::getBasicStatsHTML() const {
  std::stringstream html_object;
  std::string two_tabs{"\n\t\t"};
  std::string three_tabs{"\n\t\t\t"};
  std::string four_tabs{"\n\t\t\t\t"};
  std::string five_tabs{"\n\t\t\t\t\t"};

  // Determine font class from coverage.css and image for color bar.
  std::string font_class;
  std::string color_bar;

  if (percentage_of_called_apis < 40.f) {
    font_class = "coverNumLo";
    color_bar = "resources/ruby.png";
  } else if (percentage_of_called_apis < 80.f) {
    font_class = "coverNumMed";
    color_bar = "resources/amber.png";
  } else {
    font_class = "coverNumHi";
    color_bar = "resources/emerald.png";
  }

  if (isDeprecated()) {
    font_class = "coverDeprecated";
  }

  html_object << two_tabs << "<tr>";
  html_object << three_tabs << "<td class=\"coverFile\"><a href=\"testModules/" << group_name
              << ".html\">" << group_name << "</a></td>";
  html_object << three_tabs << "<td class=\"" << font_class << "\">" << total_number_of_apis
              << "</td>";
  html_object << three_tabs << "<td class=\"" << font_class << "\">" << number_of_api_calls
              << "</td>";
  html_object << three_tabs << "<td class=\"" << font_class << "\">" << number_of_test_cases
              << "</td>";
  html_object << three_tabs << "<td class=\"" << font_class << "\">" << called_apis.size()
              << "</td>";
  html_object << three_tabs << "<td class=\"" << font_class << "\">" << not_called_apis.size()
              << "</td>";
  html_object << three_tabs << "<td class=\"" << font_class << "\">" << deprecated_apis.size()
              << "</td>";
  html_object << three_tabs << "<td class=\"" << font_class << "\">" << std::fixed
              << std::setprecision(2) << percentage_of_called_apis << "%</td>";

  html_object << three_tabs << "<td class=\"coverBar\" align=\"center\">";
  html_object << four_tabs << "<table border=0 cellspacing=0 cellpadding=1>";
  html_object << five_tabs << "<tr><td class=\"coverBarOutline\"><img src=\"";
  html_object << color_bar << "\" width=" << percentage_of_called_apis << " height=10 alt=\""
              << percentage_of_called_apis << "%\">";
  html_object << "<img src=\"resources/snow.png\" width=" << 100.f - percentage_of_called_apis
              << " height=10 alt=\"" << percentage_of_called_apis << "\"></td></tr></table>";
  html_object << four_tabs << "</td>";
  html_object << three_tabs << "</tr>";

  return html_object.str();
}

std::string HipAPIGroup::createHTMLReport() const {
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
  html_report << one_tab << "<title>" << group_name << " Coverage report</title>";
  html_report << one_tab
              << "<link rel=\"stylesheet\" type=\"text/css\" href=\"../resources/coverage.css\">"
              << one_tab << "</head>";
  html_report << one_tab << "<body>" << one_tab
              << "<table width=\"100%\" border=0 cellspacing=0 cellpadding=0>";
  html_report << two_tabs << "<tr><td class=\"title\">" << group_name
              << " Coverage report</td></tr>";
  html_report << two_tabs
              << "<tr><td class=\"ruler\"><img src=\"../resources/glass.png\" width=3 "
                 "height=3></td></tr>\n";
  html_report << two_tabs << "<tr>" << three_tabs << "<td width=\"100%\">" << four_tabs
              << "<table cellpading=1 border=0 width=\"100%\"";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td width=\"70%\"></td>";
  html_report << six_tabs << "<td width=\"10%\" class=\"headerCovTableHead\">Value</td>";
  html_report << six_tabs << "<td width=\"20%\"></td>";
  html_report << five_tabs << "</tr>";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td class=\"headerItem\">Total number of detected HIP APIs:</td>";
  html_report << six_tabs << "<td class=\"headerCovTableEntry\">" << total_number_of_apis
              << "</td>";
  html_report << six_tabs << "<td></td>";
  html_report << five_tabs << "</tr>";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs
              << "<td class=\"headerItem\">HIP API calls within test source files:</td>";
  html_report << six_tabs << "<td class=\"headerCovTableEntry\">" << number_of_api_calls << "</td>";
  html_report << six_tabs << "<td></td>";
  html_report << five_tabs << "</tr>";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td class=\"headerItem\">Total number of test cases:</td>";
  html_report << six_tabs << "<td class=\"headerCovTableEntry\">" << number_of_test_cases
              << "</td>";
  html_report << six_tabs << "<td></td>";
  html_report << five_tabs << "</tr>";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs
              << "<td class=\"headerItem\">HIP APIs that are called at least once:</td>";
  html_report << six_tabs << "<td class=\"headerCovTableEntry\">" << called_apis.size() << "</td>";
  html_report << six_tabs << "<td></td>";
  html_report << five_tabs << "</tr>";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td class=\"headerItem\">HIP APIs that are not called at all:</td>";
  html_report << six_tabs << "<td class=\"headerCovTableEntry\">" << not_called_apis.size()
              << "</td>";
  html_report << six_tabs << "<td></td>";
  html_report << five_tabs << "</tr>";

  html_report << five_tabs << "<tr>";
  html_report << six_tabs
              << "<td class=\"headerItem\">HIP APIs that are marked as deprecated:</td>";
  html_report << six_tabs << "<td class=\"headerCovTableEntry\">" << deprecated_apis.size()
              << "</td>";
  html_report << six_tabs << "<td></td>";
  html_report << five_tabs << "</tr>";

  // Determine font class from coverage.css file based on coverage percentage.
  std::string font_class;
  if (percentage_of_called_apis < 40.f) {
    font_class = "headerCovTableEntryLo";
  } else if (percentage_of_called_apis < 80.f) {
    font_class = "headerCovTableEntryMed";
  } else {
    font_class = "headerCovTableEntryHi";
  }

  html_report << five_tabs << "<tr>";
  html_report
      << six_tabs
      << "<td class=\"headerItem\">Test coverage by implemented tests for the HIP APIs:</td>";
  html_report << six_tabs << "<td class=\"" << font_class << "\">" << std::fixed
              << std::setprecision(2) << percentage_of_called_apis << "%</td>";
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

  // Add info about Test module APIs.
  html_report << one_tab << "<center>";
  html_report << one_tab << "<table width=\"60%\" cellpadding=1 cellspacing=1 border=0>";
  html_report << two_tabs << "<tr>";
  html_report << three_tabs << "<td width=\"33%\"><br></td>";
  html_report << three_tabs << "<td width=\"33%\"</td>";
  html_report << three_tabs << "<td width=\"33%\"></td>";
  html_report << two_tabs << "</tr>";
  html_report << two_tabs << "<tr>";

  html_report << three_tabs << "<td style=\"vertical-align:top\">";
  html_report << four_tabs << "<table width=\"100%\" cellpadding=1 cellspacing=1 border=0>";
  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td class=\"tableHead\">Called APIs</td>";
  html_report << five_tabs << "</tr>";
  for (auto const& hip_api : called_apis) {
    html_report << five_tabs << "<tr>";
    html_report << six_tabs << "<td class=\"coverFile\"><a href=\"../testAPIs/" << hip_api.getName()
                << ".html\">" << hip_api.getName() << "</a></td>";
    html_report << five_tabs << "</tr>";
  }

  html_report << four_tabs << "</table>";
  html_report << three_tabs << "</td>";

  html_report << three_tabs << "<td style=\"vertical-align:top\">";
  html_report << four_tabs << "<table width=\"100%\" cellpadding=1 cellspacing=1 border=0>";
  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td class=\"tableHead\">Not called APIs</td>";
  html_report << five_tabs << "</tr>";
  for (auto const& hip_api : not_called_apis) {
    html_report << five_tabs << "<tr>";
    html_report << six_tabs << "<td class=\"coverFile\"><a href=\"../testAPIs/" << hip_api.getName()
                << ".html\">" << hip_api.getName() << "</a></td>";
    html_report << five_tabs << "</tr>";
  }
  html_report << four_tabs << "</table>";
  html_report << three_tabs << "</td>";

  html_report << three_tabs << "<td style=\"vertical-align:top\">";
  html_report << four_tabs << "<table width=\"100%\" cellpadding=1 cellspacing=1 border=0>";
  html_report << five_tabs << "<tr>";
  html_report << six_tabs << "<td class=\"tableHead\">Deprecated APIs</td>";
  html_report << five_tabs << "</tr>";
  for (auto const& hip_api : deprecated_apis) {
    html_report << five_tabs << "<tr>";
    html_report << six_tabs << "<td class=\"coverFile\"><a href=\"../testAPIs/" << hip_api.getName()
                << ".html\">" << hip_api.getName() << "</a></td>";
    html_report << five_tabs << "</tr>";
  }
  html_report << four_tabs << "</table>";
  html_report << three_tabs << "</td>";

  html_report << two_tabs << "</tr>";

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
