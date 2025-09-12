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

#include "reportGenerators.h"

BasicAPIStats::BasicAPIStats(std::vector<HipAPIGroup>& hip_api_groups)
    : number_of_called_apis{0},
      number_of_not_called_apis{0},
      number_of_deprecated_apis{0},
      total_number_of_api_calls{0},
      total_number_of_test_cases{0} {
  for (auto const& hip_api_group : hip_api_groups) {
    number_of_called_apis += hip_api_group.getNumberOfCalledAPIs();
    number_of_not_called_apis += hip_api_group.getNumberOfNotCalledAPIs();
    number_of_deprecated_apis += hip_api_group.getNumberOfDeprecatedAPIs();
    total_number_of_api_calls += hip_api_group.getTotalNumberOfCalls();
    total_number_of_test_cases += hip_api_group.getTotalNumberOfTestCases();
  }
  total_number_of_apis =
      number_of_called_apis + number_of_not_called_apis + number_of_deprecated_apis;
  tests_coverage_percentage =
      100.f * number_of_called_apis / (number_of_called_apis + number_of_not_called_apis);
}

float BasicAPIStats::getLowCoverageLimit() const { return 40.f; }

float BasicAPIStats::getMediumCoverageLimit() const { return 80.f; }

void generateXMLReportFiles(std::vector<HipAPI>& hip_apis,
                            std::vector<HipAPIGroup>& hip_api_groups) {
  BasicAPIStats basic_stats{hip_api_groups};

  std::cout << "Total number of HIP API calls: " << basic_stats.total_number_of_api_calls
            << std::endl;
  std::cout << "Number of the HIP APIs that are called at least once: "
            << basic_stats.number_of_called_apis << std::endl;
  std::cout << "Number of the HIP APIs that are not called at all: "
            << basic_stats.number_of_not_called_apis << std::endl;
  std::cout << "Number of the HIP APIs that are marked as deprecated: "
            << basic_stats.number_of_deprecated_apis << std::endl;
  std::cout
      << "Test coverage by implemented tests, for the HIP APIs that are not marked as deprecated: ";
  std::cout << basic_stats.tests_coverage_percentage << "%" << std::endl;

  /*
  Generate XML file that contains relevant information about test coverage.
  The XML file is created using raw handling of XML files, as there is no need
  for the additional 3rd party library that implements XML file CRUD operations.
  */
  std::fstream coverage_report;
  std::string report_file_name{"CoverageReport.xml"};
  coverage_report.open(report_file_name, std::ios::out);

  time_t now{time(nullptr)};
  std::string date{asctime(gmtime(&now))};
  coverage_report << "<REPORT-GENERATED-UTC>" << date << "</REPORT-GENERATED-UTC>\n";

  coverage_report << "<COVERAGE-RESULTS>\n";

  coverage_report << "\t<TOTAL-NUMBER-OF-APIs>\n\t\t<DESCRIPTION>Total number of detected HIP "
                     "APIs.</DESCRIPTION>";
  coverage_report << "\n\t\t<NUMBER>" << hip_apis.size()
                  << "</NUMBER>\n\t</TOTAL-NUMBER-OF-APIs>\n";

  coverage_report << "\t<TOTAL-NUMBER-OF-API-CALLS>\n\t\t<DESCRIPTION>Total number of HIP API "
                     "calls within test source files.</DESCRIPTION>";
  coverage_report << "\n\t\t<NUMBER>" << basic_stats.total_number_of_api_calls
                  << "</NUMBER>\n\t</TOTAL-NUMBER-OF-API-CALLS>\n";

  coverage_report << "\t<CALLED-APIs>\n\t\t<DESCRIPTION>Number of the HIP APIs that are called at "
                     "least once.</DESCRIPTION>";
  coverage_report << "\n\t\t<NUMBER>" << basic_stats.number_of_called_apis
                  << "</NUMBER>\n\t</CALLED-APIs>\n";

  coverage_report << "\t<NOT-CALLED-APIs>\n\t\t<DESCRIPTION>Number of the HIP APIs that are not "
                     "called at all.</DESCRIPTION>";
  coverage_report << "\n\t\t<NUMBER>" << basic_stats.number_of_not_called_apis
                  << "</NUMBER>\n\t</NOT-CALLED-APIs>\n";

  coverage_report << "\t<DEPRECATED-APIs>\n\t\t<DESCRIPTION>Number of the HIP APIs that are marked "
                     "as deprecated.</DESCRIPTION>";
  coverage_report << "\n\t\t<NUMBER>" << basic_stats.number_of_deprecated_apis
                  << "</NUMBER>\n\t</DEPRECATED-APIs>\n";

  coverage_report << "\t<COVERAGE-PERCENTAGE>\n\t\t<DESCRIPTION>Test coverage by implemented tests "
                     "for the HIP APIs that are not marked as deprecated.</DESCRIPTION>";
  coverage_report << "\n\t\t<VALUE>" << basic_stats.tests_coverage_percentage
                  << "%</VALUE>\n\t</COVERAGE-PERCENTAGE>";

  coverage_report << "\n</COVERAGE-RESULTS>";

  for (auto const& hip_api_group : hip_api_groups) {
    coverage_report << hip_api_group.getBasicStatsHTML();
  }

  coverage_report.close();
  std::cout << "Generated XML report file " << findAbsolutePathOfFile(report_file_name)
            << std::endl;
}

void generateHTMLReportFiles(std::vector<HipAPI>& hip_apis,
                             std::vector<HipAPIGroup>& hip_api_groups,
                             std::string tests_root_directory, std::string hipApiHeaderFile,
                             std::string hip_rtc_header_file) {
  BasicAPIStats basic_stats{hip_api_groups};

  std::fstream coverage_report;
  // Main HTML report file.
  std::string report_file_name{"./coverageReportHTML/CoverageReport.html"};
  // Directories used to store generated HTML files.
  std::string test_modules_directory{"./coverageReportHTML/testModules"};
  std::string test_apis_directory{"./coverageReportHTML/testAPIs"};
  std::filesystem::create_directories(test_modules_directory);
  std::filesystem::create_directories(test_apis_directory);

  coverage_report.open(report_file_name, std::ios::out);

  // Helper strings with tabs and newlines for better HTML formatting.
  std::string one_tab{"\n\t"};
  std::string two_tabs{"\n\t\t"};
  std::string three_tabs{"\n\t\t\t"};
  std::string four_tabs{"\n\t\t\t\t"};
  std::string five_tabs{"\n\t\t\t\t\t"};
  std::string six_tabs{"\n\t\t\t\t\t\t"};

  /*
  Create HTML file which contains report from coverage. There is no need for
  3rd party HTML libraries as the HTML report file is pretty simple and only
  consists of tables and appropriate data.
  It is better to open CoverageReport.html file in browser and view page
  source, as it is much more clear.
  */
  coverage_report << "<html lang=\"en\">";
  coverage_report << "<head>" << one_tab
                  << "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\">";
  coverage_report << one_tab << "<title>HIP API Coverage report</title>";
  coverage_report << one_tab
                  << "<link rel=\"stylesheet\" type=\"text/css\" href=\"resources/coverage.css\">"
                  << one_tab << "</head>";
  coverage_report << one_tab << "<body>" << one_tab
                  << "<table width=\"100%\" border=0 cellspacing=0 cellpadding=0>";
  coverage_report << two_tabs << "<tr><td class=\"title\">HIP API Coverage report</td></tr>";
  coverage_report
      << two_tabs
      << "<tr><td class=\"ruler\"><img src=\"resources/glass.png\" width=3 height=3></td></tr>\n";
  coverage_report << two_tabs << "<tr>" << three_tabs << "<td width=\"100%\">" << four_tabs
                  << "<table cellpading=1 border=0 width=\"100%\"";

  coverage_report << five_tabs << "<tr>";
  coverage_report << six_tabs
                  << "<td width=\"20%\" class=\"headerItem\">Catch2 tests location:</td>";
  coverage_report << six_tabs << "<td width=\"30%\" class=\"headerValue\">" << tests_root_directory
                  << "</td>";
  coverage_report << six_tabs << "<td width=\"20%\"></td>";
  coverage_report << six_tabs << "<td width=\"10%\" class=\"headerCovTableHead\">Value</td>";
  coverage_report << six_tabs << "<td width=\"20%\"></td>";
  coverage_report << five_tabs << "</tr>";

  coverage_report << five_tabs << "<tr>";
  coverage_report << six_tabs << "<td class=\"headerItem\">Source files included:</td>";
  coverage_report << six_tabs << "<td class=\"headerValue\">" << hipApiHeaderFile << "</td>";
  coverage_report << six_tabs << "<td class=\"headerItem\">Total number of detected HIP APIs:</td>";
  coverage_report << six_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.total_number_of_apis << "</td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << five_tabs << "</tr>";

  coverage_report << five_tabs << "<tr>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << six_tabs << "<td class=\"headerValue\">" << hip_rtc_header_file << "</td>";
  coverage_report << six_tabs
                  << "<td class=\"headerItem\">HIP API calls within test source files:</td>";
  coverage_report << six_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.total_number_of_api_calls << "</td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << five_tabs << "</tr>";

  coverage_report << five_tabs << "<tr>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << six_tabs << "<td class=\"headerItem\">Total number of test cases:</td>";
  coverage_report << six_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.total_number_of_test_cases << "</td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << five_tabs << "</tr>";

  coverage_report << five_tabs << "<tr>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << six_tabs
                  << "<td class=\"headerItem\">HIP APIs that are called at least once:</td>";
  coverage_report << six_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.number_of_called_apis << "</td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << five_tabs << "</tr>";

  coverage_report << five_tabs << "<tr>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << six_tabs
                  << "<td class=\"headerItem\">HIP APIs that are not called at all:</td>";
  coverage_report << six_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.number_of_not_called_apis << "</td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << five_tabs << "</tr>";

  coverage_report << five_tabs << "<tr>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << six_tabs
                  << "<td class=\"headerItem\">HIP APIs that are marked as deprecated:</td>";
  coverage_report << six_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.number_of_deprecated_apis << "</td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << five_tabs << "</tr>";

  // Based on the tests coverage percentage, pick a color for displaying it.
  std::string font_class;
  if (basic_stats.tests_coverage_percentage < basic_stats.getLowCoverageLimit()) {
    font_class = "headerCovTableEntryLo";
  } else if (basic_stats.tests_coverage_percentage < basic_stats.getMediumCoverageLimit()) {
    font_class = "headerCovTableEntryMed";
  } else {
    font_class = "headerCovTableEntryHi";
  }

  coverage_report << five_tabs << "<tr>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report
      << six_tabs
      << "<td class=\"headerItem\">Test coverage by implemented tests for the HIP APIs:</td>";
  coverage_report << six_tabs << "<td class=\"" << font_class << "\">" << std::fixed
                  << std::setprecision(2) << basic_stats.tests_coverage_percentage << "%</td>";
  coverage_report << six_tabs << "<td></td>";
  coverage_report << five_tabs << "</tr>";

  coverage_report << five_tabs
                  << "<tr><td><img src=\"resources/glass.png\" width=3 height=3></td></tr>";
  coverage_report << four_tabs << "</table>";
  coverage_report << three_tabs << "</td>";
  coverage_report << two_tabs << "</tr>";

  coverage_report
      << two_tabs
      << "<tr><td class=\"ruler\"><img src=\"resources/glass.png\" width=3 height=3></td></tr>\n";
  coverage_report << one_tab << "</table>";

  // Add info about HIP API Groups.
  coverage_report << one_tab << "<center>";
  coverage_report << one_tab << "<table width=\"60%\" cellpadding=1 cellspacing=1 border=0>";
  coverage_report << two_tabs << "<tr>";
  coverage_report << three_tabs << "<td width=\"30%\"><br></td>";
  coverage_report << three_tabs << "<td width=\"10%\"></td>";
  coverage_report << three_tabs << "<td width=\"10%\"></td>";
  coverage_report << three_tabs << "<td width=\"10%\"></td>";
  coverage_report << three_tabs << "<td width=\"10%\"></td>";
  coverage_report << three_tabs << "<td width=\"10%\"></td>";
  coverage_report << three_tabs << "<td width=\"10%\"></td>";
  coverage_report << three_tabs << "<td width=\"5%\"></td>";
  coverage_report << three_tabs << "<td width=\"5%\"></td>";
  coverage_report << two_tabs << "</tr>";

  coverage_report << two_tabs << "<tr>";
  coverage_report << three_tabs << "<td class=\"tableHead\">Module</td>";
  coverage_report << three_tabs << "<td class=\"tableHead\">HIP APIs</td>";
  coverage_report << three_tabs << "<td class=\"tableHead\">HIP API Calls</td>";
  coverage_report << three_tabs << "<td class=\"tableHead\">Test cases</td>";
  coverage_report << three_tabs << "<td class=\"tableHead\">Called APIs</td>";
  coverage_report << three_tabs << "<td class=\"tableHead\">Not called APIs</td>";
  coverage_report << three_tabs << "<td class=\"tableHead\">Deprecated APIs</td>";
  coverage_report << three_tabs << "<td class=\"tableHead\" colspan=2>Coverage</td>";
  coverage_report << two_tabs << "</tr>";

  /*
  Get basic stats for each API Group in HTML format and append it to the main HTML.
  Create an HTML page for each API Group for more detailed information, as they are
  used as hyperlinks from the main HTML page.
  */
  for (auto const& hip_api_group : hip_api_groups) {
    if (hip_api_group.getTotalNumberOfAPIs() == 0) {
      std::cout << "[SKIP_FROM_COV] Couldn't detect any valid HIP API within module "
                << hip_api_group.getName() << " hence skipping its HTML generation" << std::endl;
      continue;
    }
    coverage_report << hip_api_group.getBasicStatsHTML();

    std::fstream coverage_module_report;
    std::string report_module_file_name{test_modules_directory + "/" + hip_api_group.getName() +
                                        ".html"};
    coverage_module_report.open(report_module_file_name, std::ios::out);
    coverage_module_report << hip_api_group.createHTMLReport();
    coverage_module_report.close();
  }

  coverage_report << two_tabs << "<tr>";
  coverage_report << three_tabs << "<td></td>";
  coverage_report << three_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.total_number_of_apis << "</td>";
  coverage_report << three_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.total_number_of_api_calls << "</td>";
  coverage_report << three_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.total_number_of_test_cases << "</td>";
  coverage_report << three_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.number_of_called_apis << "</td>";
  coverage_report << three_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.number_of_not_called_apis << "</td>";
  coverage_report << three_tabs << "<td class=\"headerCovTableEntry\">"
                  << basic_stats.number_of_deprecated_apis << "</td>";
  coverage_report << three_tabs << "<td class=\"headerCovTableEntry\">" << std::fixed
                  << std::setprecision(2) << basic_stats.tests_coverage_percentage << "%</td>";
  coverage_report << three_tabs << "<td></td>";
  coverage_report << two_tabs << "</tr>";

  coverage_report << one_tab << "</table>";
  coverage_report << one_tab << "</center>";
  coverage_report << one_tab << "<br>";

  coverage_report << one_tab << "<table width=\"100%\" border=0 cellspacing=0 cellpadding=0>";
  coverage_report
      << two_tabs
      << "<tr><td class=\"ruler\"><img src=\"resources/glass.png\" width=3 height=3></td></tr>";
  coverage_report << one_tab << "</table>";
  coverage_report << one_tab << "<br>";

  coverage_report << one_tab << "<center>";
  coverage_report << one_tab << "<table width=\"30%\" cellpadding=1 cellspacing=1 border=0>";
  coverage_report << two_tabs << "<tr>";
  coverage_report << three_tabs << "<td width=\"3%\"></td>";
  coverage_report << three_tabs << "<td width=\"97%\"></td>";
  coverage_report << two_tabs << "</tr>";

  coverage_report << two_tabs << "<tr>";
  coverage_report << three_tabs << "<td class=\"tableHead\" colspan=2>Color legend</td>";
  coverage_report << two_tabs << "</tr>";

  coverage_report << two_tabs << "<tr>";
  coverage_report << three_tabs << "<td class=\"coverDeprecated\"></td>";
  coverage_report << three_tabs << "<td class=\"coverFile\">Module is deprecated</td>";
  coverage_report << two_tabs << "</tr>";

  coverage_report << two_tabs << "<tr>";
  coverage_report << three_tabs << "<td class=\"coverNumLo\"></td>";
  coverage_report
      << three_tabs
      << "<td class=\"coverFile\">Percentage of called APIs within a module is less than 40%</td>";
  coverage_report << two_tabs << "</tr>";

  coverage_report << two_tabs << "<tr>";
  coverage_report << three_tabs << "<td class=\"coverNumMed\"></td>";
  coverage_report << three_tabs
                  << "<td class=\"coverFile\">Percentage of called APIs within a module is between "
                     "40% and 80%</td>";
  coverage_report << two_tabs << "</tr>";

  coverage_report << two_tabs << "<tr>";
  coverage_report << three_tabs << "<td class=\"coverNumHi\"></td>";
  coverage_report << three_tabs
                  << "<td class=\"coverFile\">Percentage of called APIs within a module is larger "
                     "than 80%</td>";
  coverage_report << two_tabs << "</tr>";

  coverage_report << one_tab << "</table>";
  coverage_report << one_tab << "</center>";
  coverage_report << one_tab << "<br>";

  coverage_report << one_tab << "<table width=\"100%\" border=0 cellspacing=0 cellpadding=0>";
  coverage_report
      << two_tabs
      << "<tr><td class=\"ruler\"><img src=\"resources/glass.png\" width=3 height=3></td></tr>";

  time_t now{time(nullptr)};
  std::string date{asctime(gmtime(&now))};
  coverage_report << two_tabs << "<tr><td class=\"versionInfo\">Generated: " << date;
  coverage_report << two_tabs << " UTC</td></tr>";
  coverage_report << one_tab << "</table>";
  coverage_report << one_tab << "<br>";
  coverage_report << "\n</body>\n</html>";

  coverage_report.close();

  // Create HTML report for each API, as they are used as hyperlinks from Groups HTML.
  for (auto const& hip_api : hip_apis) {
    std::fstream coverage_api_report;
    std::string report_api_file_name{test_apis_directory + "/" + hip_api.getName() + ".html"};
    coverage_api_report.open(report_api_file_name, std::ios::out);
    coverage_api_report << hip_api.createHTMLReport();
    coverage_api_report.close();
  }

  std::cout << "Generated HTML report file " << findAbsolutePathOfFile(report_file_name)
            << std::endl;
}
