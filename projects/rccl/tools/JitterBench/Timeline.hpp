#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>

struct TimelineData
{
  std::string rowLabel;
  std::string barLabel;
  std::string toolTip;
  uint64_t    startTime;
  uint64_t    stopTime;
};

void ExportToTimeLine(std::string outputFilename,
                      std::string rowLabelName,
                      std::string barLabelName,
                      std::vector<TimelineData> const& timelineData)
{
  FILE *fp = fopen(outputFilename.c_str(), "w");

  fprintf(fp, "<script type=\"text/javascript\" src=\"https://www.gstatic.com/charts/loader.js\"></script>\n");
  fprintf(fp, "<script type=\"text/javascript\">\n");
  fprintf(fp, "google.charts.load(\"current\", {packages:[\"timeline\"]});\n");
  fprintf(fp, "google.charts.setOnLoadCallback(drawChart);\n");
  fprintf(fp, "\n");
  fprintf(fp, "function drawChart() {\n");
  fprintf(fp, "  var container = document.getElementById('myTimeline');\n");
  fprintf(fp, "  var chart = new google.visualization.Timeline(container);\n");
  fprintf(fp, "  var dataTable = new google.visualization.DataTable();\n");
  fprintf(fp, "\n");
  fprintf(fp, "  dataTable.addColumn({ type: 'string', id:   '%s' });\n", rowLabelName.c_str());
  fprintf(fp, "  dataTable.addColumn({ type: 'string', id:   '%s' });\n", barLabelName.c_str());
  fprintf(fp, "  dataTable.addColumn({ type: 'string', role: 'tooltip'});\n");
  fprintf(fp, "  dataTable.addColumn({ type: 'number', id:   'Start' });\n");
  fprintf(fp, "  dataTable.addColumn({ type: 'number', id:   'End' });\n");
  fprintf(fp, "  dataTable.addRows([\n");

  for (int i = 0; i < timelineData.size(); i++)
  {
    TimelineData const& t = timelineData[i];
    fprintf(fp, "   [ '%s', '%s', '%s', %lu, %lu ]%s\n", t.rowLabel.c_str(),
            t.barLabel.c_str(), t.toolTip.c_str(), t.startTime, t.stopTime, i + 1 == timelineData.size() ? "]);" : ",");
  }

  fprintf(fp, "  chart.draw(dataTable);\n");
  fprintf(fp, "}\n");
  fprintf(fp, "</script>\n");
  fprintf(fp, "<div id=\"myTimeline\" style=\"width: 100%%; height: 100%%;\"></div>\n");
  fclose(fp);
}
