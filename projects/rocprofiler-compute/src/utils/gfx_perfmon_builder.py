##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import sys

from lxml import html
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QTreeView,
    QWidget,
    qApp,
)


# class view(QWidget):
class mainWindow(QMainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()

        ###############################################################################
        # SOC Parameters
        ##############################################################################

        # Per IP block max number of simulutaneous counters
        # GFX IP Blocks
        self.perfmon_config = {
            "SQ": 8,
            "TA": 2,
            "TD": 2,
            "TCP": 4,
            "TCC": 4,
            "CPC": 2,
            "CPF": 2,
            "SPI": 2,
            "GRBM": 2,
            "GDS": 4,
        }

        # GFX Architectures
        self.soc_arch_list = ["gfx906", "gfx908", "gfx90a"]

        ###############################################################################
        # Window layout Design
        ##############################################################################

        self.block_list = []
        self.nodes_dict = {}  # list of QStandardItem

        self.tree = QTreeView(self)
        self.table = QTableWidget()

        # Tree setup
        self.tree.header().setDefaultSectionSize(180)
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Metric", "Block", "Event", "Definition"])

        self.tree.setModel(self.model)
        # self.importData(data)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Set up click processing
        self.tree.clicked.connect(self.pmc_select)
        # self.tree.expandAll()

        # Table setup
        tableHeader = list(self.perfmon_config.keys())
        self.table.setColumnCount(len(tableHeader))
        self.table.setHorizontalHeaderLabels(tableHeader)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.showGrid()

        self.setWindowTitle("GFX Perfmon Builder")
        # layout: lhs: metrics; rhs: selected perfmon
        layout = QHBoxLayout(self)
        layout.addWidget(self.tree)
        layout.addWidget(self.table)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Add Status
        self.statusBar()

        ###############################################################################
        # Window Menu Design
        ##############################################################################

        # Setup file menu
        menuBar = self.menuBar()
        menuBar.setNativeMenuBar(False)

        openAction = QAction("&Open", self)
        openAction.setShortcut("Ctrl+O")
        openAction.setStatusTip("Open GFX Metrics file")
        openAction.triggered.connect(self.openGFXDialog)

        saveAction = QAction("&Save", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.setStatusTip("Save to PMC file")
        saveAction.triggered.connect(self.exportGFXDialog)

        exitAction = QAction("&Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit")
        exitAction.triggered.connect(self.close)

        # Create new action
        fileMenu = menuBar.addMenu("&File")
        fileMenu.addActions([openAction, saveAction])
        fileMenu.addSeparator()
        fileMenu.addActions([exitAction])

    def openGFXDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open GFX Metrics", "", "XML Files (*.xml)", "XML(*.xml)"
        )

        # Parse the xml
        if fileName:
            xmlparsed = html.parse(fileName)
            self.importData(xmlparsed)

    def exportGFXDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Export PMC Counters", "", "Text File (*.txt)", "Text File(*.txt)"
        )

        # Parse the xml
        if fileName:
            self.exportPMCCounters(fileName)

    def exportPMCCounters(self, fileName):
        f = open(fileName, "w")

        total_IP_blocks = len(list(self.perfmon_config.keys()))
        for row in range(self.table.rowCount()):
            pmc_str = "pmc: "
            for col in range(total_IP_blocks):
                cell = self.table.item(row, col)

                if cell:
                    pmc_str = pmc_str + "  ".join(cell.text().split("\n")) + "  "

            f.write(pmc_str + "\n")

        # Add standard lines
        f.write("\n\n")
        f.write("gpu: \n")
        f.write("dispatch: \n")
        f.write("kernel: \n")

        f.close()

        return

    def pmc_metric_selected(self, metric_name, col):
        # check if the metric already exists
        metric_selected = False

        for row in range(self.table.rowCount()):
            entry = self.table.item(row, col)
            if entry:
                pmc_list = entry.text().split(sep="\n")
                if metric_name in pmc_list:
                    metric_selected = True
                    break

        return metric_selected

    def pmc_remove_metric(self, metric_name, IP_block):
        # Remove the metric to pmc table, if it is selected

        # Map SQC to SQ, since they share the same Perfmon block
        if IP_block == "SQC":
            IP_block = "SQ"

        # not action if it is for a ghost IP!
        if not IP_block in list(self.perfmon_config.keys()):
            return

        # This is the column we need to add/remove perfmon counters
        col = list(self.perfmon_config.keys()).index(IP_block)

        if not self.pmc_metric_selected(metric_name, col):
            return

        pmc_list = []
        for row in range(self.table.rowCount()):
            entry = self.table.item(row, col)

            if entry:
                pmc_list = pmc_list + entry.text().split(sep="\n")
                # clear the cell, we will re-allocate the pmc

            self.table.takeItem(row, col)

        # allowed PMC counters per batch
        max_pmc_num = self.perfmon_config[IP_block]

        # remote this metric and re-segment the list and refill all rows in this column
        pmc_list.remove(metric_name)

        # We are empty now, do nothing
        if len(pmc_list) == 0:
            return

        for row in range((len(pmc_list) + max_pmc_num - 1) // max_pmc_num):
            start_index = row * max_pmc_num
            pmc_str = "\n".join(pmc_list[start_index : start_index + max_pmc_num])
            self.table.setItem(row, col, QTableWidgetItem(pmc_str))

        # Remove last row, if empty
        last_row = self.table.rowCount() - 1
        empty_row = True
        total_cols = len(list(self.perfmon_config.keys()))
        for cindex in range(total_cols):
            x = self.table.item(last_row, cindex)

            if x and x.text():
                empty_row = False
                break

        if empty_row:
            self.table.removeRow(last_row)

    def pmc_add_metric(self, metric_name, IP_block):
        # Add the metric to pmc table, if not there yet

        # Map SQC to SQ, since they share the same Perfmon block
        if IP_block == "SQC":
            IP_block = "SQ"

        if not IP_block in list(self.perfmon_config.keys()):
            return

        # This is the column we need to add/remove perfmon counters
        col = list(self.perfmon_config.keys()).index(IP_block)

        # check if the metric already exists
        if self.pmc_metric_selected(metric_name, col):
            return

        # metric is not bucket yet, add it!
        if self.table.rowCount() == 0:
            # starting from scratch!
            self.table.insertRow(0)
            self.table.setItem(0, col, QTableWidgetItem(metric_name))
            return

        # find the row to insert
        for row in range(self.table.rowCount()):
            entry = self.table.item(row, col)
            if not entry:
                # print("search insert pos, row:", row, ", cell empty")
                break

            if len(entry.text().split(sep="\n")) < self.perfmon_config[IP_block]:
                # print("found")
                break

        entry = self.table.item(row, col)
        if not entry:
            # put it into the empty cell
            self.table.setItem(row, col, QTableWidgetItem(metric_name))
            return

        pmc_list = entry.text().split(sep="\n")

        if len(pmc_list) < self.perfmon_config[IP_block]:
            # we still have hit per-IP HW counters limit, add it to the last row
            pmc_list.append(metric_name)
            pmc_str = "\n".join(pmc_list)
            self.table.setItem(row, col, QTableWidgetItem(pmc_str))
            self.table.resizeRowsToContents()
        else:
            # Start a new row
            row = row + 1
            self.table.insertRow(row)
            self.table.setItem(row, col, QTableWidgetItem(metric_name))

    def pmc_select(self, item):
        metric_name = item.data()
        if (
            not metric_name in self.nodes_dict
            or not self.nodes_dict[metric_name].isCheckable()
        ):
            return

        # only proper metrics check/uncheck is processed here.
        IP_block = item.data().split(sep="_")[0]

        if self.nodes_dict[metric_name].checkState() == 0:
            # unselect the metric in the table if it is currently selected
            self.pmc_remove_metric(metric_name, IP_block)

        elif self.nodes_dict[metric_name].checkState() == 2:
            self.pmc_add_metric(metric_name, IP_block)

    # Function to save populate treeview with a dictionary
    def importData(self, xmlparsed, root=None):
        self.model.setRowCount(0)
        if root is None:
            root = self.model.invisibleRootItem()

        for x in xmlparsed.getiterator():
            # Add SoC node to Root
            if x.tag in self.soc_arch_list:
                parent = root
                parent.appendRow([QStandardItem(x.tag)])
                self.nodes_dict[x.tag] = parent.child(parent.rowCount() - 1)

            # check all metrics in an SoC family
            if x.tag == "metric" and x.getparent().tag in self.soc_arch_list:
                # New IP block (e.g., SQ), detected, create a new hierarchy for the block
                if not x.attrib["block"] in self.block_list:
                    self.block_list.append(x.attrib["block"])
                    parent = self.nodes_dict[x.getparent().tag]  # the SoC node
                    parent.appendRow(
                        [
                            QStandardItem(x.attrib["block"]),
                            QStandardItem(""),
                            QStandardItem(""),
                            QStandardItem(""),
                        ]
                    )

                    # record the tree node for the block
                    self.nodes_dict[x.attrib["block"]] = parent.child(
                        parent.rowCount() - 1
                    )

                # Add metric node to the Block node
                parent = self.nodes_dict[x.attrib["block"]]
                metric_name = QStandardItem(x.attrib["name"])
                metric_name.setCheckable(True)
                parent.appendRow(
                    [
                        metric_name,
                        QStandardItem(x.attrib["block"]),
                        QStandardItem(x.attrib["event"]),
                        QStandardItem(x.attrib["descr"]),
                    ]
                )

                self.nodes_dict[x.attrib["name"]] = parent.child(parent.rowCount() - 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # populate the view with GFX metrics.xml
    window = mainWindow()

    # show the view
    window.setGeometry(300, 100, 600, 300)
    # view.setWindowTitle('GFX Perfmon Counters')
    window.show()

    # start the application
    sys.exit(app.exec_())
