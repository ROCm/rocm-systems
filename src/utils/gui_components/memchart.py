##############################################################################
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

from dash import html
from dash_svg import G, Path, Rect, Svg, Text

from utils.logger import console_error


def insert_chart_data(mem_data, base_data):
    if len(mem_data) != 1:
        console_error("Memory Chart config doesn't follow expected formatting")

    table_config = mem_data[0]["metric_table"]

    original_df = base_data.dfs[table_config["id"]]

    display_columns = original_df.columns.values.tolist().copy()
    display_df = original_df[display_columns]

    alias = display_df["Metric"].values
    values = display_df["Value"].values

    memchart_values = {}
    for i in range(0, len(alias)):
        memchart_values[alias[i]] = values[i]

    return G(
        className="data",
        children=[
            # ----------------------------------------
            # Instr Buff Block
            # TODO: double check wave_occupancy
            Text(
                x="52",
                y="313",
                id="wave_occ",
                fill="#FFFF33",
                fontSize="20px",
                fontWeight="bold",
                children=memchart_values["Wavefront Occupancy"],
            ),
            Text(
                x="49",
                y="394",
                id="wave_life",
                fill="#FFFF33",
                fontSize="20px",
                fontWeight="bold",
                children=memchart_values["Wave Life"],
            ),
            # ----------------------------------------
            # Instr Dispatch Block
            Text(
                x="386",
                y="46",
                id="salu",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=format_value_for_display(memchart_values["SALU"]),
            ),
            Text(
                x="386",
                y="96",
                id="smem",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=format_value_for_display(memchart_values["SMEM"]),
            ),
            Text(
                x="386",
                y="146",
                id="valu",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=format_value_for_display(memchart_values["VALU"]),
            ),
            Text(
                x="386",
                y="196",
                id="mfma",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=format_value_for_display(memchart_values["MFMA"]),
            ),
            Text(
                x="386",
                y="245",
                id="vmem",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=format_value_for_display(memchart_values["VMEM"]),
            ),
            Text(
                x="386",
                y="296",
                id="lds",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["LDS"],
            ),
            Text(
                x="386",
                y="344",
                id="gws",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["GWS"],
            ),
            Text(
                x="386",
                y="396",
                id="br",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["BR"],
            ),
            # ----------------------------------------
            # Exec Block
            Text(
                x="480",
                y="99",
                id="active_cu",
                fill="#FFFF33",
                fontSize="20px",
                fontWeight="bold",
                children=memchart_values["Active CUs"],
            ),  # x=454
            Text(
                x="580",
                y="154",
                id="vgpr",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["VGPR"],
            ),
            Text(
                x="581",
                y="183",
                id="sgpr",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["SGPR"],
            ),
            Text(
                x="580",
                y="226",
                id="lds_alloc",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["LDS Allocation"],
            ),
            Text(
                x="580",
                y="255",
                id="scratch_alloc",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["Scratch Allocation"],
            ),
            Text(
                x="580",
                y="298",
                id="wavefronts",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["Wavefronts"],
            ),
            Text(
                x="580",
                y="328",
                id="workgroups",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["Workgroups"],
            ),
            # ----------------------------------------
            # LDS Block
            Text(
                x="723",
                y="78",
                id="lds_req",
                fill="#FFFFFF",
                fontSize="12px",
                children=memchart_values["LDS Req"],
            ),
            Text(
                x="839",
                y="85",
                id="lds_util",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["LDS Util"],
            ),
            Text(
                x="839",
                y="117",
                id="lds_lat",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["LDS Latency"],
            ),
            # ----------------------------------------
            # Vector L1 Cache Block
            Text(
                x="708",
                y="204",
                id="vl1_rd",
                fill="#FFFFFF",
                fontSize="12px",
                children=format_value_for_display(memchart_values["VL1 Rd"]),
            ),
            Text(
                x="708",
                y="233",
                id="vl1_wr",
                fill="#FFFFFF",
                fontSize="12px",
                children=format_value_for_display(memchart_values["VL1 Wr"]),
            ),
            Text(
                x="716",
                y="265",
                id="vl1_atom",
                fill="#FFFFFF",
                fontSize="12px",
                children=memchart_values["VL1 Atomic"],
            ),
            Text(
                x="840",
                y="193",
                id="vl1_hit",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["VL1 Hit"],
            ),
            Text(
                x="840",
                y="224",
                id="vl1_lat",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["VL1 Lat"],
            ),
            Text(
                x="840",
                y="256",
                id="vl1_coales",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["VL1 Coalesce"],
            ),
            Text(
                x="838",
                y="288",
                id="vl1_stall",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["VL1 Stall"],
            ),
            Text(
                x="1000",
                y="203",
                id="vl1_l2_rd",
                fill="#FFFFFF",
                fontSize="12px",
                children=format_value_for_display(memchart_values["VL1_L2 Rd"]),
            ),
            Text(
                x="1000",
                y="232",
                id="vl1_l2_wr",
                fill="#FFFFFF",
                fontSize="12px",
                children=format_value_for_display(memchart_values["VL1_L2 Wr"]),
            ),
            Text(
                x="1008",
                y="264",
                id="vl1_l2_atom",
                fill="#FFFFFF",
                fontSize="12px",
                children=memchart_values["VL1_L2 Atomic"],
            ),
            # ----------------------------------------
            # Scalar L1D Cache Block
            Text(
                x="709",
                y="384",
                id="sl1_rd",
                fill="#FFFFFF",
                fontSize="12px",
                children=memchart_values["sL1D Rd"],
            ),
            Text(
                x="838",
                y="372",
                id="sl1_hit",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["sL1D Hit"],
            ),
            Text(
                x="838",
                y="404",
                id="sl1_lat",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["sL1D Lat"],
            ),
            Text(
                x="1000",
                y="351",
                id="sl1_l2_rd",
                fill="#FFFFFF",
                fontSize="12px",
                children=memchart_values["sL1D_L2 Rd"],
            ),
            Text(
                x="1000",
                y="380",
                id="sl1_l2_wr",
                fill="#FFFFFF",
                fontSize="12px",
                children=memchart_values["sL1D_L2 Wr"],
            ),
            Text(
                x="1008",
                y="412",
                id="sl1_l2_atom",
                fill="#FFFFFF",
                fontSize="12px",
                children=memchart_values["sL1D_L2 Atomic"],
            ),
            # ----------------------------------------
            # Instr L1  Cache Block
            Text(
                x="492",
                y="498",
                id="il1_fetch",
                fill="#FFFFFF",
                fontSize="12px",
                children=memchart_values["IL1 Fetch"],
            ),
            Text(
                x="837",
                y="491",
                id="il1_hit",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["IL1 Hit"],
            ),
            Text(
                x="837",
                y="522",
                id="il1_lat",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["IL1 Lat"],
            ),
            Text(
                x="1015",
                y="500",
                id="il1_l2_req",
                fill="#FFFFFF",
                fontSize="12px",
                children=format_value_for_display(memchart_values["IL1_L2 Rd"]),
            ),
            # ----------------------------------------
            # L2 Cache Block(inside)
            Text(
                x="1145",
                y="213",
                id="l2_rd",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=format_value_for_display(memchart_values["L2 Rd"]),
            ),
            Text(
                x="1145",
                y="238",
                id="l2_wr",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=format_value_for_display(memchart_values["L2 Wr"]),
            ),
            Text(
                x="1145",
                y="264",
                id="l2_atom",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["L2 Atomic"],
            ),
            Text(
                x="1145",
                y="292",
                id="l2_hit",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["L2 Hit"],
            ),
            Text(
                x="1145",
                y="356",
                id="l2_rd_lat",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["L2 Rd Lat"],
            ),
            Text(
                x="1145",
                y="382",
                id="l2_wr_lat",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["L2 Wr Lat"],
            ),
            # ----------------------------------------
            # Fabric Block
            Text(
                x="1317",
                y="243",
                id="l2_fabric_rd",
                fill="#FFFFFF",
                fontSize="12px",
                children=format_value_for_display(memchart_values["Fabric_L2 Rd"]),
            ),
            Text(
                x="1317",
                y="272",
                id="l2_fabric_wr",
                fill="#FFFFFF",
                fontSize="12px",
                children=format_value_for_display(memchart_values["Fabric_L2 Wr"]),
            ),
            Text(
                x="1319",
                y="303",
                id="l2_fabric_atom",
                fill="#FFFFFF",
                fontSize="12px",
                children=memchart_values["Fabric_L2 Atomic"],
            ),
            Text(
                x="1435",
                y="285",
                id="fabric_rd_lat",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["Fabric Rd Lat"],
            ),
            Text(
                x="1435",
                y="310",
                id="fabric_wr_lat",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["Fabric Wr Lat"],
            ),
            Text(
                x="1435",
                y="336",
                id="fabric_atom_lat",
                fill="rgb(0, 0, 0)",
                fontSize="12px",
                children=memchart_values["Fabric Atomic Lat"],
            ),
            Text(
                x="1578",
                y="240",
                id="hbm_rd",
                fill="#FFFFFF",
                fontSize="12px",
                children=format_value_for_display(memchart_values["HBM Rd"]),
            ),
            Text(
                x="1577",
                y="269",
                id="hbm_wr",
                fill="#FFFFFF",
                fontSize="12px",
                children=format_value_for_display(memchart_values["HBM Wr"]),
            ),
        ],
    )


def get_memchart(mem_data, base_data):
    return html.Section(
        id="memchart",
        children=[
            html.Div(
                id="memchart-svg",
                children=[
                    Svg(
                        children=[
                            G(
                                children=[
                                    G(
                                        className="instr-buff",
                                        children=[
                                            Rect(x="30", y="25.5"),
                                            Rect(x="20", y="32"),
                                            Rect(x="10", y="42"),
                                            Rect(x="0", y="52"),
                                        ],
                                    ),
                                    G(
                                        className="fabric",
                                        children=[
                                            Rect(x="1373", y="213"),
                                            Rect(x="1363", y="203"),
                                        ],
                                    ),
                                    G(
                                        className="cache",
                                        children=[
                                            Rect(x="757", y="460"),
                                            Rect(x="757", y="345"),
                                            Rect(id="a3", x="757", y="32"),
                                            Rect(id="b3", x="757", y="165"),
                                        ],
                                    ),
                                    G(
                                        className="fabric-connections",
                                        children=[
                                            Rect(x="1383", y="56"),
                                            Rect(x="1383", y="451"),
                                            Rect(x="1606.69", y="227.43"),
                                        ],
                                    ),
                                    G(
                                        className="inner-inst-buff",
                                        children=[
                                            Rect(x="20", y="92"),
                                            Rect(x="20", y="170.28"),
                                        ],
                                    ),
                                    G(
                                        className="misc-rec",
                                        children=[
                                            Rect(x="1063", y="32"),
                                            Rect(id="a6", x="326.25", y="138.25"),
                                            Rect(id="b6", x="450", y="122"),
                                        ],
                                    ),
                                    G(
                                        className="val-1",
                                        children=[
                                            Rect(x="319", y="32", rx="3", ry="3"),
                                            Rect(x="319", y="82", rx="3", ry="3"),
                                            Rect(x="319", y="132", rx="3", ry="3"),
                                            Rect(x="319", y="182", rx="3", ry="3"),
                                            Rect(x="319", y="231", rx="3", ry="3"),
                                            Rect(x="319", y="282", rx="3", ry="3"),
                                            Rect(x="319", y="329.5", rx="3", ry="3"),
                                            Rect(x="319", y="382", rx="3", ry="3"),
                                            Rect(x="1367.69", y="271", rx="3", ry="3"),
                                            Rect(
                                                x="1367.69", y="296.5", rx="3", ry="3"
                                            ),
                                            Rect(
                                                x="1367.69", y="322.5", rx="3", ry="3"
                                            ),
                                            Rect(x="1078", y="199", rx="3", ry="3"),
                                            Rect(x="1078", y="224.5", rx="3", ry="3"),
                                            Rect(x="1078", y="250.5", rx="3", ry="3"),
                                            Rect(x="771.44", y="103", rx="3", ry="3"),
                                            Rect(
                                                x="770.44", y="358.75", rx="3", ry="3"
                                            ),
                                            Rect(
                                                x="770.44", y="390.25", rx="3", ry="3"
                                            ),
                                            Rect(x="769.44", y="477", rx="3", ry="3"),
                                            Rect(x="769.44", y="508.5", rx="3", ry="3"),
                                            Rect(x="1078", y="278", rx="3", ry="3"),
                                            Rect(x="1078", y="342.5", rx="3", ry="3"),
                                            Rect(x="1078", y="368.5", rx="3", ry="3"),
                                            Rect(x="772.44", y="179", rx="3", ry="3"),
                                            Rect(
                                                x="772.44", y="210.18", rx="3", ry="3"
                                            ),
                                            Rect(x="771.44", y="71.28", rx="3", ry="3"),
                                            Rect(x="772.44", y="242", rx="3", ry="3"),
                                            Rect(x="770.44", y="274.5", rx="3", ry="3"),
                                        ],
                                    ),
                                    G(
                                        className="val-2",
                                        children=[
                                            Rect(x="362", y="32", rx="3", ry="3"),
                                            Rect(x="362", y="82", rx="3", ry="3"),
                                            Rect(x="362", y="132", rx="3", ry="3"),
                                            Rect(x="362", y="182", rx="3", ry="3"),
                                            Rect(x="362", y="231", rx="3", ry="3"),
                                            Rect(x="362", y="282", rx="3", ry="3"),
                                            Rect(x="362", y="329.5", rx="3", ry="3"),
                                            Rect(x="362", y="382", rx="3", ry="3"),
                                        ],
                                    ),
                                    G(
                                        className="val-3",
                                        children=[
                                            Rect(x="1410.69", y="271", rx="3", ry="3"),
                                            Rect(
                                                x="1410.69", y="296.5", rx="3", ry="3"
                                            ),
                                            Rect(
                                                x="1410.69", y="322.5", rx="3", ry="3"
                                            ),
                                            Rect(x="1121", y="199", rx="3", ry="3"),
                                            Rect(x="1121", y="224.5", rx="3", ry="3"),
                                            Rect(x="1121", y="250.5", rx="3", ry="3"),
                                            Rect(x="814.44", y="103", rx="3", ry="3"),
                                            Rect(
                                                x="813.44", y="358.75", rx="3", ry="3"
                                            ),
                                            Rect(
                                                x="813.44", y="390.25", rx="3", ry="3"
                                            ),
                                            Rect(x="812.44", y="477", rx="3", ry="3"),
                                            Rect(x="812.44", y="508.5", rx="3", ry="3"),
                                            Rect(x="1121", y="278", rx="3", ry="3"),
                                            Rect(x="1121", y="342.5", rx="3", ry="3"),
                                            Rect(x="1121", y="368.5", rx="3", ry="3"),
                                            Rect(x="815.44", y="179", rx="3", ry="3"),
                                            Rect(
                                                x="815.44", y="210.18", rx="3", ry="3"
                                            ),
                                            Rect(x="814.44", y="71.28", rx="3", ry="3"),
                                            Rect(x="815.44", y="242", rx="3", ry="3"),
                                            Rect(x="813.44", y="274.5", rx="3", ry="3"),
                                        ],
                                    ),
                                    G(
                                        className="val-4",
                                        children=[
                                            Rect(x="460", y="212.5", rx="3", ry="3"),
                                            Rect(x="460", y="241", rx="3", ry="3"),
                                            Rect(x="460", y="284.54", rx="3", ry="3"),
                                            Rect(x="460", y="314", rx="3", ry="3"),
                                            Rect(x="460", y="140.32", rx="3", ry="3"),
                                            Rect(x="460", y="169.16", rx="3", ry="3"),
                                        ],
                                    ),
                                    G(
                                        className="val-5",
                                        children=[
                                            Rect(
                                                x="548.25",
                                                y="212.98",
                                                rx="2.86",
                                                ry="2.86",
                                            ),
                                            Rect(
                                                x="548.25",
                                                y="241.48",
                                                rx="2.86",
                                                ry="2.86",
                                            ),
                                            Rect(
                                                x="548.25",
                                                y="285.02",
                                                rx="2.86",
                                                ry="2.86",
                                            ),
                                            Rect(
                                                x="548.25",
                                                y="314.48",
                                                rx="2.86",
                                                ry="2.86",
                                            ),
                                            Rect(
                                                x="548.25",
                                                y="140.8",
                                                rx="2.86",
                                                ry="2.86",
                                            ),
                                            Rect(
                                                x="549.5",
                                                y="169.64",
                                                rx="2.86",
                                                ry="2.86",
                                            ),
                                        ],
                                    ),
                                    G(
                                        className="lines-arrows",
                                        children=[
                                            Path(
                                                id="p1",
                                                d="M 100 243.72 L 120 220.28 L "
                                                  "475 220.28 L 495 243.72 Z",
                                                fill="#ffffff",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 260 62 L 285.99 62.15",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 260 112 L 285.15 111.92",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 260 162 L 285.57 161.69",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 260 212 L 285.15 211.85",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 285.66 262.41 L 260 262.07",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 260 312 L 284.73 312.18",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 260 362 L 284.28 361.95",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 260 412 L 285.57 412.12",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 310.02 62.15 L 413.63 62.01",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 418.88 62 L 411.89 65.51 L "
                                                  "413.63 62.01 L 411.88 58.51 Z",
                                                fill="#ff8000",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 309.92 111.92 L 413.63 112",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 418.88 112 L 411.88 115.49 L "
                                                  "413.63 112 L 411.88 108.49 Z",
                                                fill="#ff8000",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 309.08 162.08 L 413.63 162",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 418.88 162 L 411.88 165.51 L "
                                                  "413.63 162 L 411.88 158.51 Z",
                                                fill="#ff8000",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 310 212 L 413.63 212",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 418.88 212 L 411.88 215.5 "
                                                "L 413.63 212 L 411.88 208.5 Z",
                                                fill="#ff8000",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 309.92 262.02 L 413.63 262",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 418.88 262 L 411.88 265.5 L "
                                                "413.63 262 L 411.88 258.5 Z",
                                                fill="#ff8000",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 309.36 312.18 L 413.63 312.01",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 418.88 312 L 411.89 315.51 L "
                                                "413.63 312.01 L 411.88 308.51 Z",
                                                fill="#ff8000",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 309.08 361.95 L 413.63 362",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 418.88 362 L 411.88 365.5 L "
                                                "413.63 362 L 411.88 358.5 Z",
                                                fill="#ff8000",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 309.36 408.56 L 413.63 408.97",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 418.88 409 L 411.87 412.47 L "
                                                "413.63 408.97 L 411.9 405.47 Z",
                                                fill="#ff8000",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 750 207 L 652.37 207",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 647.12 207 L 654.12 203.5 L "
                                                "652.37 207 L 654.12 210.5 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 646 236.57 L 743.63 236.03",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 748.88 236.01 L 741.9 239.54 L "
                                                "743.63 236.03 L 741.86 232.54 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 750 502 L 110 502 Q 100 "
                                                "502 100 492 L 100 468.37",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 100 463.12 L 103.5 470.12 L "
                                                "100 468.37 L 96.5 470.12 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1050 504 L 942.37 504",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 937.12 504 L 944.12 500.5 L "
                                                "942.37 504 L 944.12 507.5 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1597.69 242.93 L 1534.06 242.93",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1528.81 242.93 L 1535.81 239.43 L "
                                                "1534.06 242.93 L 1535.81 246.43 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1527.69 274.64 L 1591.32 274.64",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1596.57 274.64 L 1589.57 278.14 L "
                                                "1591.32 274.64 L 1589.57 271.14 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1413 196 L 1413 132.37",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1413 127.12 L 1416.5 134.12 L "
                                                "1413 132.37 L 1409.5 134.12 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1443 126 L 1443 189.63",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1443 194.88 L 1439.5 187.88 L "
                                                "1443 189.63 L 1446.5 187.88 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1413.36 441 L 1413.03 377.37",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1413.01 372.12 L 1416.54 379.1 L "
                                                "1413.03 377.37 L 1409.54 379.14 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1443 371 L 1443.33 432.13",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1443.35 437.38 L 1439.82 430.4 L "
                                                "1443.33 432.13 L 1446.82 430.36 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1145.25 341.38 L 1141.75 334.38 L "
                                                "1145.25 336.13 L 1148.75 334.38 Z",
                                                fill="rgb(0, 0, 0)",
                                                stroke="rgb(0, 0, 0)",
                                            ),
                                            Path(
                                                d="M 740 82 L 652.37 82",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 647.12 82 L 654.12 78.5 L 652.37 "
                                                "82 L 654.12 85.5 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 744 386.75 L 656.37 386.75",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 651.12 386.75 L 658.12 383.25 L "
                                                "656.37 386.75 L 658.12 390.25 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 651.37 269 L 743.63 269",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 646.12 269 L 653.12 265.5 L "
                                                "651.37 269 L 653.12 272.5 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 748.88 269 L 741.88 272.5 L "
                                                "743.63 269 L 741.88 265.5 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1042 206.41 L 944.37 206.41",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 939.12 206.41 L 946.12 202.91 L "
                                                "944.37 206.41 L 946.12 209.91 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 938 235.98 L 1035.63 235.44",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1040.88 235.42 L 1033.9 238.95 L "
                                                "1035.63 235.44 L 1033.86 231.95 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 943.37 268.41 L 1035.63 268.41",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 938.12 268.41 L 945.12 264.91 L "
                                                "943.37 268.41 L 945.12 271.91 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1040.88 268.41 L 1033.88 271.91 L "
                                                "1035.63 268.41 L 1033.88 264.91 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1042 354.32 L 944.37 354.32",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 939.12 354.32 L 946.12 350.82 L "
                                                "944.37 354.32 L 946.12 357.82 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 938 383.89 L 1035.63 383.35",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1040.88 383.33 L 1033.9 386.86 L "
                                                "1035.63 383.35 L 1033.86 379.86 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 943.37 416.32 L 1035.63 416.32",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 938.12 416.32 L 945.12 412.82 L "
                                                "943.37 416.32 L 945.12 419.82 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1040.88 416.32 L 1033.88 419.82 L "
                                                "1035.63 416.32 L 1033.88 412.82 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1355 245.75 L 1257.37 245.75",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1252.12 245.75 L 1259.12 242.25 L "
                                                "1257.37 245.75 L 1259.12 249.25 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1251 275.32 L 1348.63 274.78",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1353.88 274.76 L 1346.9 278.29 L "
                                                "1348.63 274.78 L 1346.86 271.29 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1256.37 307.75 L 1348.63 307.75",
                                                fill="none",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1251.12 307.75 L 1258.12 304.25 L "
                                                "1256.37 307.75 L 1258.12 311.25 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                d="M 1353.88 307.75 L 1346.88 311.25 L "
                                                "1348.63 307.75 L 1346.88 304.25 Z",
                                                fill="#00cccc",
                                                stroke="#00cccc",
                                            ),
                                            Path(
                                                id="p2",
                                                d="M 235 67 L 245 57 L "
                                                "265 57 L 275 67 Z",
                                                fill="#ffffff",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 220 56 L 250 56",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 210 65 L 250 65",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 200 74 L 250 74",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 230 47.5 L 250 47.5",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                id="p3",
                                                d="M 235 117 L 245 107 L 265 "
                                                "107 L 275 117 Z",
                                                fill="#ffffff",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 220 106 L 250 106",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 210 115 L 250 115",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 200 124 L 250 124",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 230 97.5 L 250 97.5",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                id="p4",
                                                d="M 235 167 L 245 157 L "
                                                "265 157 L 275 167 Z",
                                                fill="#ffffff",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 220 156 L 250 156",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 210 165 L 250 165",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 200 174 L 250 174",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 230 147.5 L 250 147.5",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                id="p5",
                                                d="M 235 217 L 245 207 L "
                                                "265 207 L 275 217 Z",
                                                fill="#ffffff",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 220 206 L 250 206",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 210 215 L 250 215",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 200 224 L 250 224",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 230 197.5 L 250 197.5",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                id="p6",
                                                d="M 235 267 L 245 257 L "
                                                "265 257 L 275 267 Z",
                                                fill="#ffffff",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 220 256 L 250 256",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 210 265 L 250 265",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 200 274 L 250 274",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 230 247.5 L 250 247.5",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                id="p7",
                                                d="M 235 317 L 245 307 L "
                                                "265 307 L 275 317 Z",
                                                fill="#ffffff",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 220 306 L 250 306",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 210 315 L 250 315",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 200 324 L 250 324",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 230 297.5 L 250 297.5",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                id="p8",
                                                d="M 235 367 L 245 357 L "
                                                "265 357 L 275 367 Z",
                                                fill="#ffffff",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 220 356 L 250 356",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 210 365 L 250 365",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 200 374 L 250 374",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 230 347.5 L 250 347.5",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                id="p9",
                                                d="M 235 417 L 245 407 L "
                                                "265 407 L 275 417 Z",
                                                fill="#ffffff",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 220 406 L 250 406",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 210 415 L 250 415",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                            Path(
                                                d="M 200 424 L 250 424",
                                                fill="none",
                                                stroke="#ff8000",
                                            ),
                                        ],
                                    ),
                                    G(
                                        className="labels",
                                        children=[
                                            Text(
                                                x="12",
                                                y="278",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                children="Wave Occupancy",
                                            ),
                                            Text(
                                                x="12",
                                                y="363",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                children="Wave Life",
                                            ),
                                            Text(
                                                x="1428",
                                                y="80",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                textAnchor="middle",
                                                children="xGMI /",
                                            ),
                                            Text(
                                                x="1428",
                                                y="105",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                textAnchor="middle",
                                                children="PCIe",
                                            ),
                                            Text(
                                                x="1428",
                                                y="487",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                textAnchor="middle",
                                                children="GMI",
                                            ),
                                            Text(
                                                x="1652",
                                                y="263",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                textAnchor="middle",
                                                children="HBM",
                                            ),
                                            Text(
                                                x="1438",
                                                y="230",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                textAnchor="middle",
                                                children="Fabric",
                                            ),
                                            Text(
                                                x="360",
                                                y="47",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="SALU:",
                                            ),
                                            Text(
                                                x="360",
                                                y="97",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="SMEM:",
                                            ),
                                            Text(
                                                x="360",
                                                y="147",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="VALU:",
                                            ),
                                            Text(
                                                x="360",
                                                y="197",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="MFMA:",
                                            ),
                                            Text(
                                                x="360",
                                                y="246",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="VMEM:",
                                            ),
                                            Text(
                                                x="360",
                                                y="297",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="LDS:",
                                            ),
                                            Text(
                                                x="360",
                                                y="344",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="GWS:",
                                            ),
                                            Text(
                                                x="360",
                                                y="397",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Br:",
                                            ),
                                            Text(
                                                x="1463",
                                                y="285",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="cycles",
                                            ),
                                            Text(
                                                x="1408",
                                                y="286",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Rd:",
                                            ),
                                            Text(
                                                x="1463",
                                                y="310",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="cycles",
                                            ),
                                            Text(
                                                x="1408",
                                                y="311",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Wr:",
                                            ),
                                            Text(
                                                x="1463",
                                                y="336",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="cycles",
                                            ),
                                            Text(
                                                x="1408",
                                                y="337",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Atomic:",
                                            ),
                                            Text(
                                                x="1118",
                                                y="214",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Rd:",
                                            ),
                                            Text(
                                                x="1118",
                                                y="239",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Wr:",
                                            ),
                                            Text(
                                                x="1118",
                                                y="265",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Atomic:",
                                            ),
                                            Text(
                                                x="867",
                                                y="117",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="cycles",
                                            ),
                                            Text(
                                                x="812",
                                                y="117",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Lat:",
                                            ),
                                            Text(
                                                x="866",
                                                y="372",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="%",
                                            ),
                                            Text(
                                                x="810",
                                                y="373",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Hit:",
                                            ),
                                            Text(
                                                x="866",
                                                y="404",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="cycles",
                                            ),
                                            Text(
                                                x="810",
                                                y="405",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Lat:",
                                            ),
                                            Text(
                                                x="865",
                                                y="491",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="%",
                                            ),
                                            Text(
                                                x="809",
                                                y="492",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Hit:",
                                            ),
                                            Text(
                                                x="865",
                                                y="522",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="cycles",
                                            ),
                                            Text(
                                                x="809",
                                                y="523",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Lat:",
                                            ),
                                            Text(
                                                x="1556",
                                                y="239",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Rd:",
                                            ),
                                            Text(
                                                x="1554",
                                                y="269",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Wr:",
                                            ),
                                            Text(
                                                x="699",
                                                y="77",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Req:",
                                            ),
                                            Text(
                                                x="684",
                                                y="204",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Rd:",
                                            ),
                                            Text(
                                                x="684",
                                                y="233",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Wr:",
                                            ),
                                            Text(
                                                x="696",
                                                y="265",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Atomic:",
                                            ),
                                            Text(
                                                x="102",
                                                y="312",
                                                fill="#FFFF33",
                                                fontSize="20px",
                                                fontWeight="bold",
                                                children="per-GCD",
                                            ),
                                            Text(
                                                x="102",
                                                y="393",
                                                fill="#FFFF33",
                                                fontSize="20px",
                                                fontWeight="bold",
                                                children="cycles",
                                            ),
                                            Text(
                                                x="1173",
                                                y="292",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="%",
                                            ),
                                            Text(
                                                x="1118",
                                                y="293",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Hit:",
                                            ),
                                            Text(
                                                x="1173",
                                                y="356",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="cycles",
                                            ),
                                            Text(
                                                x="1118",
                                                y="357",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Rd:",
                                            ),
                                            Text(
                                                x="1173",
                                                y="382",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="cycles",
                                            ),
                                            Text(
                                                x="1118",
                                                y="383",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Wr:",
                                            ),
                                            Text(
                                                x="32",
                                                y="126",
                                                fill="#FFFFFF",
                                                fontSize="14px",
                                                children="Wave 0 Instr buff",
                                            ),
                                            Text(
                                                x="32",
                                                y="205",
                                                fill="#FFFFFF",
                                                fontSize="14px",
                                                children="Wave N-1 Instr buff",
                                            ),
                                            Text(
                                                x="442",
                                                y="69",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                children="Active CUs",
                                            ),
                                            Text(
                                                x="868",
                                                y="193",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="%",
                                            ),
                                            Text(
                                                x="812",
                                                y="194",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Hit:",
                                            ),
                                            Text(
                                                x="868",
                                                y="224",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="cycles",
                                            ),
                                            Text(
                                                x="812",
                                                y="225",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Lat:",
                                            ),
                                            Text(
                                                x="867",
                                                y="85",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="%",
                                            ),
                                            Text(
                                                x="812",
                                                y="85",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Util:",
                                            ),
                                            Text(
                                                x="868",
                                                y="256",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="%",
                                            ),
                                            Text(
                                                x="813",
                                                y="256",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Coales:",
                                            ),
                                            Text(
                                                x="432",
                                                y="18",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                children="Exec",
                                            ),
                                            Text(
                                                x="12",
                                                y="18",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                children="Instr Buff",
                                            ),
                                            Text(
                                                x="250",
                                                y="18",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                children="Instr Dispatch",
                                            ),
                                            Text(
                                                x="761",
                                                y="26",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                children="LDS",
                                            ),
                                            Text(
                                                x="760",
                                                y="158",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                children="Vector L1 Cache",
                                            ),
                                            Text(
                                                x="761",
                                                y="337",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                children="Scalar L1D Cache",
                                            ),
                                            Text(
                                                x="761",
                                                y="451",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                children="Instr L1 Cache",
                                            ),
                                            Text(
                                                x="1153",
                                                y="63",
                                                fill="#FFFFFF",
                                                fontSize="20px",
                                                textAnchor="middle",
                                                children="L2 Cache",
                                            ),
                                            Text(
                                                x="991",
                                                y="499",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Req:",
                                            ),
                                            Text(
                                                x="866",
                                                y="288",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                children="%",
                                            ),
                                            Text(
                                                x="811",
                                                y="288",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Stall:",
                                            ),
                                            Text(
                                                x="468",
                                                y="497",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Fetch:",
                                            ),
                                            Text(
                                                x="1153",
                                                y="333",
                                                fill="#FFFFFF",
                                                fontSize="14px",
                                                textAnchor="middle",
                                                textDecoration="underline",
                                                children="Latency",
                                            ),
                                            Text(
                                                x="543",
                                                y="227",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="LDS Alloc:",
                                            ),
                                            Text(
                                                x="543",
                                                y="255",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Scratch Alloc:",
                                            ),
                                            Text(
                                                x="543",
                                                y="299",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Wavefronts:",
                                            ),
                                            Text(
                                                x="543",
                                                y="328",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Workgroups:",
                                            ),
                                            Text(
                                                x="543",
                                                y="155",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="VGPRs:",
                                            ),
                                            Text(
                                                x="544",
                                                y="183",
                                                fill="rgb(0, 0, 0)",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="SGPRs:",
                                            ),
                                            Text(
                                                x="684",
                                                y="384",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Rd:",
                                            ),
                                            Text(
                                                x="976",
                                                y="204",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Rd:",
                                            ),
                                            Text(
                                                x="976",
                                                y="232",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Wr:",
                                            ),
                                            Text(
                                                x="988",
                                                y="264",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Atomic:",
                                            ),
                                            Text(
                                                x="976",
                                                y="352",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Rd:",
                                            ),
                                            Text(
                                                x="976",
                                                y="380",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Wr:",
                                            ),
                                            Text(
                                                x="988",
                                                y="412",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Atomic:",
                                            ),
                                            Text(
                                                x="1292",
                                                y="243",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Rd:",
                                            ),
                                            Text(
                                                x="1293",
                                                y="272",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Wr:",
                                            ),
                                            Text(
                                                x="1301",
                                                y="304",
                                                fill="#FFFFFF",
                                                fontSize="12px",
                                                textAnchor="end",
                                                children="Atomic:",
                                            ),
                                        ],
                                    ),
                                    insert_chart_data(mem_data, base_data),
                                ]
                            )
                        ],
                        viewBox="-0.5 -0.5 1698 543",
                    )
                ],
            )
        ],
    )


def format_value_for_display(value, max_length=6):
    """
    Format values to prevent overflow in SVG text elements.
    """
    #####
    # TODO: this is quick fix to prevent value overflow.
    # The long term solution should be dynamically adjust
    # SVG dimensions and positions to maintain visual
    # integrity while preventing overflow.
    #####

    # 1. If non-numerical
    if isinstance(value, str):
        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass  # Keep as string
    # 2. If numerical
    if isinstance(value, (int, float)):
        value = abs(value)
        if value >= 1000000000:
            value = f"{value / 1000000000:.1f}B"
        elif value >= 1000000:
            value = f"{value / 1000000:.1f}M"
        elif value >= 1000:
            value = f"{value / 1000:.1f}K"
        elif value == int(value):
            value = str(int(value))
        else:
            value = f"{value:.1f}"
    else:
        value = str(value)

    # 3. Truncate if needed
    if len(value) > max_length:
        value = value[: max_length - 1] + "…"

    return value
