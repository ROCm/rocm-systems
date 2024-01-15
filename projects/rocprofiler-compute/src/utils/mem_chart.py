###############################################################################
# MIT License
#
# Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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
###############################################################################

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Generator
from types import SimpleNamespace as NS
from plotille import Canvas


# A basic rect frame for any block or group of wires where all its elements should
# be within this range, except: (a) the label(title) might be on the top of it,
# (b) some wires around it don't have to be grouped specifically.
@dataclass
class RectFrame:
    label: str
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 1.0
    y_max: float = 1.0


# Instr Buff Block
@dataclass
class InstrBuff(RectFrame):
    wave_occupancy: int = -1
    wave_life: int = -1

    def draw(self, canvas):
        # print("---------", self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min, self.y_max + 1.0, self.label)

        canvas.rect(self.x_min, self.y_min, self.x_max - 2.0, self.y_max - 1.0)
        canvas.rect(
            self.x_min + 1.0, self.y_min + 0.5, self.x_max - 1.0, self.y_max - 0.5
        )
        canvas.rect(self.x_min + 2.0, self.y_min + 1.0, self.x_max, self.y_max)

        canvas.rect(
            self.x_min + 4.0, self.y_max - 3.5, self.x_max - 4.0, self.y_max - 2.0
        )
        canvas.text(self.x_min + 5.0, self.y_max - 3.0, r"Wave   0 Instr Buf")

        canvas.rect(
            self.x_min + 4.0, self.y_max - 7.5, self.x_max - 4.0, self.y_max - 6.0
        )
        canvas.text(self.x_min + 5.0, self.y_max - 7.0, r"Wave N-1 Instr Buf")

        canvas.text(self.x_min + 7.0, self.y_min + 5.0, r"Wave Occupancy")
        canvas.text(
            self.x_min + 10.0,
            self.y_min + 4.0,
            "{val:>3.0f}  per CU".format(val=self.wave_occupancy),
            color="yellow",
        )
        canvas.text(self.x_min + 7.0, self.y_min + 3.0, r"Wave Life")
        canvas.text(
            self.x_min + 8.0,
            self.y_min + 2.0,
            "{val:>5.0f}  cycles".format(val=self.wave_life),
            color="yellow",
        )


# Wires between Instr Buff and Instr Dispatch
@dataclass
class Wire_InstrBuff_InstrDispatch(RectFrame):
    def draw(self, canvas):
        # Todo: finer wires for connections
        canvas.line(self.x_min + 2, self.y_min, self.x_min + 2, self.y_max)
        canvas.line(self.x_max, self.y_min + 1.5, self.x_max, self.y_max - 1.5)
        canvas.line(self.x_min + 2, self.y_min, self.x_max, self.y_min + 1.5)
        canvas.line(self.x_min + 2, self.y_max, self.x_max - 0.5, self.y_max - 1.5)


# Instr Dispatch Block
@dataclass
class InstrDispatch(RectFrame):
    top_rect_x_min: float = 0.0
    top_rect_x_max: float = 0.0
    top_rect_y_min: float = 0.0
    top_rect_y_max: float = 0.0
    text_x_offset: float = 1.0
    text_y_offset: float = 0.5
    line_y_offset: float = 0.5
    rect_y_offset: float = 3.0
    instrs: Dict[str, int] = field(default_factory=dict)

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)

        self.top_rect_x_min = self.x_min + 2.0
        self.top_rect_x_max = self.top_rect_x_min + 14.0
        self.top_rect_y_min = self.y_max - 1.5
        self.top_rect_y_max = self.y_max

        i = 0
        for k, v in self.instrs.items():
            # print(k,v)
            text = "{key:<6}: {val:>4.0f}".format(key=k, val=v)
            canvas.text(
                self.top_rect_x_min + self.text_x_offset,
                self.top_rect_y_min - self.rect_y_offset * i + self.text_y_offset,
                text,
            )
            canvas.text(
                self.top_rect_x_min - 2,
                self.top_rect_y_min - self.rect_y_offset * i,
                "------------------>",
            )
            i = i + 1


# Exec Block
@dataclass
class Exec(RectFrame):
    active_cus: int = 0
    num_cus: int = 0
    vgprs: int = 0
    sgprs: int = 0
    lds_alloc: int = 0
    scratch_alloc: int = 0
    wavefronts: int = 0
    workgroups: int = 0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)

        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 2.0, self.y_max - 2.0, "Active CUs")
        canvas.text(
            self.x_min + 2.0,
            self.y_max - 3.0,
            "{a:>3.0f}/{n:>3.0f}".format(a=self.active_cus, n=self.num_cus),
            color="yellow",
        )

        canvas.rect(
            self.x_min + 2.0, self.y_max - 7.0, self.x_max - 2.0, self.y_max - 5.0
        )
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 6.0,
            "{key:<6}: {val:>5}".format(key="VGPRs", val=self.vgprs),
        )

        canvas.rect(
            self.x_min + 2.0, self.y_max - 10.0, self.x_max - 2.0, self.y_max - 8.0
        )
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 9.0,
            "{key:<6}: {val:>5.0f}".format(key="SGPRs", val=self.sgprs),
        )

        canvas.rect(
            self.x_min + 2.0, self.y_max - 15.0, self.x_max - 2.0, self.y_max - 12.0
        )
        canvas.text(self.x_min + 4.0, self.y_max - 13.0, "LDS Alloc:")
        canvas.text(
            self.x_min + 4.0, self.y_max - 14.0, "{val:>13.0f}".format(val=self.lds_alloc)
        )

        canvas.rect(
            self.x_min + 2.0, self.y_max - 19.0, self.x_max - 2.0, self.y_max - 16.0
        )
        canvas.text(self.x_min + 4.0, self.y_max - 17.0, "Scratch Alloc:")
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 18.0,
            "{val:>13.0f}".format(val=self.scratch_alloc),
        )

        canvas.rect(
            self.x_min + 2.0, self.y_max - 24.0, self.x_max - 2.0, self.y_max - 21.0
        )
        canvas.text(self.x_min + 4.0, self.y_max - 22.0, "Wavefronts:")
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 23.0,
            "{val:>13.0f}".format(val=self.wavefronts),
        )

        canvas.rect(
            self.x_min + 2.0, self.y_max - 28.0, self.x_max - 2.0, self.y_max - 25.0
        )
        canvas.text(self.x_min + 4.0, self.y_max - 26.0, "Workgroups:")
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 27.0,
            "{val:>13.0f}".format(val=self.workgroups),
        )


# Wires between Exec block and GDS, LDS, Vector L1 cache, Scalar L1D Cache
@dataclass
class Wire_E_GLVS(RectFrame):
    text_x_offset: float = 3.0

    lds_req: int = -1
    vl1_rd: int = -1
    vl1_wr: int = -1
    vl1_atomic: int = -1
    sl1_rd: int = -1

    def draw(self, canvas):
        canvas.text(
            self.x_min + self.text_x_offset,
            self.y_max - 2.0,
            "{key:<6}: {val:4.0f}".format(key="Req", val=self.lds_req),
        )
        canvas.text(
            self.x_min + self.text_x_offset - 2, self.y_max - 3.0, "<---------------"
        )

        canvas.text(
            self.x_min + self.text_x_offset,
            self.y_max - 10.0,
            "{key:<6}: {val:>4.0f}".format(key="Rd", val=self.vl1_rd),
        )
        canvas.text(
            self.x_min + self.text_x_offset - 2, self.y_max - 11.0, "<---------------"
        )
        canvas.text(
            self.x_min + self.text_x_offset,
            self.y_max - 12.0,
            "{key:<6}: {val:>4.0f}".format(key="Wt", val=self.vl1_wr),
        )
        canvas.text(
            self.x_min + self.text_x_offset - 2, self.y_max - 13.0, "--------------->"
        )
        canvas.text(
            self.x_min + self.text_x_offset,
            self.y_max - 14.0,
            "{key:<6}: {val:>4.0f}".format(key="Atomic", val=self.vl1_atomic),
        )
        canvas.text(
            self.x_min + self.text_x_offset - 2, self.y_max - 15.0, "<-------------->"
        )

        canvas.text(
            self.x_min + self.text_x_offset,
            self.y_max - 22.0,
            "{key:<6}: {val:>4.0f}".format(key="Rd", val=self.sl1_rd),
        )
        canvas.text(
            self.x_min + self.text_x_offset - 2, self.y_max - 23.0, "<---------------"
        )


# Wire between Instr Buff and Instr L1 Cache
@dataclass
class Wire_InstrBuff_IL1Cache(RectFrame):
    il1_fetch: int = 0

    def draw(self, canvas):
        end_col = int(self.y_max - self.y_min)
        canvas.text(self.x_min, self.y_max - 1, "^")
        for i in range(2, end_col):
            canvas.text(self.x_min, self.y_max - i, "|")
        canvas.text(
            self.x_min + 27,
            self.y_max - end_col + 1,
            "{key:<6}: {val:>4.0f}".format(key="Fetch", val=self.il1_fetch),
        )
        canvas.text(
            self.x_min, self.y_max - end_col, "-" * (int(self.x_max - self.x_min))
        )


# GDS Block
@dataclass
class GDS(RectFrame):
    gws: int = -1
    latency: int = -1

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.rect(
            self.x_min + 2.0, self.y_min + 2.5, self.x_max - 2.0, self.y_max - 1.0
        )
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 2.0,
            "{key:<4}: {val:>4.0f} cycles".format(key="GWS", val=self.gws),
        )

        canvas.rect(
            self.x_min + 2.0, self.y_min + 0.5, self.x_max - 2.0, self.y_min + 2.0
        )
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 4.0,
            "{key:<4}: {val:>4.0f} cycles".format(key="Lat", val=self.latency),
        )


# LDS Block
@dataclass
class LDS(RectFrame):
    util: int = -1
    latency: int = -1

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(
            self.x_min + 2.0,
            self.y_max - 2.0,
            "{key:<6}: {val:>6.0f} %".format(key="Util", val=self.util),
        )
        canvas.text(
            self.x_min + 2.0,
            self.y_max - 4.0,
            "{key:<6}: {val:>6.0f} cycles".format(key="Lat", val=self.latency),
        )


# Vector L1 Cache Block
@dataclass
class VectorL1Cache(RectFrame):
    hit: int = -1
    latency: int = -1
    coales: int = -1
    stall: int = -1

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.text(
            self.x_min + 2.0,
            self.y_max - 2.0,
            "{key:<6}: {val:>6.0f} %".format(key="Hit", val=self.hit),
        )
        canvas.text(
            self.x_min + 2.0,
            self.y_max - 4.0,
            "{key:<6}: {val:>6.0f} cycles".format(key="Lat", val=self.latency),
        )
        canvas.text(
            self.x_min + 2.0,
            self.y_max - 6.0,
            "{key:<6}: {val:>6.0f} %".format(key="Coales", val=self.coales),
        )
        canvas.text(
            self.x_min + 2.0,
            self.y_max - 8.0,
            "{key:<6}: {val:>6.0f} cycles".format(key="Stall", val=self.stall),
        )


# Scalar L1D Cache
@dataclass
class ScalarL1DCache(RectFrame):
    hit: int = -1
    latency: int = -1

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.text(
            self.x_min + 2.0,
            self.y_max - 2.0,
            "{key:<6}: {val:>6} %".format(key="Hit", val=self.hit),
        )
        canvas.text(
            self.x_min + 2.0,
            self.y_max - 4.0,
            "{key:<6}: {val:>6} cycles".format(key="Lat", val=self.latency),
        )


# Instr L1 Cache
@dataclass
class InstrL1Cache(RectFrame):
    hit: int = -1
    latency: int = -1

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.text(
            self.x_min + 2.0,
            self.y_max - 2.0,
            "{key:<6}: {val:>6.0f} %".format(key="Hit", val=self.hit),
        )
        canvas.text(
            self.x_min + 2.0,
            self.y_max - 4.0,
            "{key:<6}: {val:>6} cycles".format(key="Lat", val=self.latency),
        )


# Wires between Vector L1 cache, Scalar L1D Cache, Instr L1 cache and L2 Cache
@dataclass
class Wires_L1_L2(RectFrame):
    text_v_x_offset: float = 0.0

    vl1_l2_rd: int = -1
    vl1_l2_wr: int = -1
    vl1_l2_atomic: int = -1
    sl1_l2_rd: int = -1
    sl1_l2_wr: int = -1
    sl1_l2_atomic: int = -1
    il1_l2_req: int = -1

    def draw(self, canvas):
        canvas.text(
            self.x_min + self.text_v_x_offset,
            self.y_max - 2.0,
            "{key:<6}: {val:>4.0f}".format(key="Rd", val=self.vl1_l2_rd),
        )
        canvas.text(
            self.x_min + self.text_v_x_offset - 2, self.y_max - 3.0, "<---------------"
        )
        canvas.text(
            self.x_min + self.text_v_x_offset,
            self.y_max - 4.0,
            "{key:<6}: {val:>4.0f}".format(key="Wr", val=self.vl1_l2_wr),
        )
        canvas.text(
            self.x_min + self.text_v_x_offset - 2, self.y_max - 5.0, "--------------->"
        )
        canvas.text(
            self.x_min + self.text_v_x_offset,
            self.y_max - 6.0,
            "{key:<6}: {val:>4.0f}".format(key="Atomic", val=self.vl1_l2_atomic),
        )
        canvas.text(
            self.x_min + self.text_v_x_offset - 2, self.y_max - 7.0, "<-------------->"
        )

        canvas.text(
            self.x_min,
            self.y_max - 12.0,
            "{key:<6}: {val:>4.0f}".format(key="Rd", val=self.sl1_l2_rd),
        )
        canvas.text(self.x_min - 2, self.y_max - 13.0, "<---------------")
        canvas.text(
            self.x_min,
            self.y_max - 14.0,
            "{key:<6}: {val:>4.0f}".format(key="Wr", val=self.sl1_l2_wr),
        )
        canvas.text(self.x_min - 2, self.y_max - 15.0, "--------------->")
        canvas.text(
            self.x_min,
            self.y_max - 16.0,
            "{key:<6}: {val:>4.0f}".format(key="Atomic", val=self.sl1_l2_atomic),
        )
        canvas.text(self.x_min - 2, self.y_max - 17.0, "<-------------->")

        canvas.text(
            self.x_min,
            self.y_max - 22.0,
            "{key:<6}: {val:>4.0f}".format(key="Req", val=self.il1_l2_req),
        )
        canvas.text(self.x_min - 2, self.y_max - 23.0, "<---------------")


# L2 Cache
@dataclass
class L2Cache(RectFrame):
    rd: int = -1
    wr: int = -1
    atomic: int = -1
    hit: int = -1
    rd_lat: int = -1
    wr_lat: int = -1

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.rect(
            self.x_min + 2.0, self.y_max - 5.0, self.x_max - 2.0, self.y_max - 3.0
        )
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 4.0,
            "{key:<6}: {val:>6.0f} %".format(key="Hit", val=self.hit),
        )

        canvas.text(self.x_min + 2.0, self.y_max - 7.0, "Request")
        canvas.rect(
            self.x_min + 2.0, self.y_max - 16.0, self.x_max - 2.0, self.y_max - 7.5
        )
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 10.0,
            "{key:<6}: {val:>6.0f}".format(key="Rd", val=self.rd),
        )
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 12.0,
            "{key:<6}: {val:>6.0f}".format(key="Wr", val=self.wr),
        )
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 14.0,
            "{key:<6}: {val:>6.0f}".format(key="Atomic", val=self.atomic),
        )

        canvas.text(self.x_min + 2.0, self.y_max - 19.0, "Latency (cycles)")
        canvas.rect(
            self.x_min + 2.0, self.y_max - 25.0, self.x_max - 2.0, self.y_max - 19.5
        )

        canvas.text(
            self.x_min + 4.0,
            self.y_max - 22.0,
            "{key:<6}: {val:>6.0f}".format(key="Rd", val=self.rd_lat),
        )
        canvas.text(
            self.x_min + 4.0,
            self.y_max - 24.0,
            "{key:<6}: {val:>6.0f}".format(key="Wr", val=self.wr_lat),
        )


# Wires between L2 block and Fabric
@dataclass
class Wire_L2_Fabric(RectFrame):
    text_x_offset: float = 3.0

    rd: int = -1
    wr: int = -1
    atomic: int = -1

    def draw(self, canvas):
        canvas.text(
            self.x_min + self.text_x_offset,
            self.y_max - 2.0,
            "{key:<6}: {val:>4.0f}".format(key="Rd", val=self.rd),
        )
        canvas.text(
            self.x_min + self.text_x_offset - 2, self.y_max - 3.0, "<---------------"
        )
        canvas.text(
            self.x_min + self.text_x_offset,
            self.y_max - 4.0,
            "{key:<6}: {val:>4.0f}".format(key="Wr", val=self.wr),
        )
        canvas.text(
            self.x_min + self.text_x_offset - 2, self.y_max - 5.0, "--------------->"
        )
        canvas.text(
            self.x_min + self.text_x_offset,
            self.y_max - 6.0,
            "{key:<6}: {val:>4.0f}".format(key="Atomic", val=self.atomic),
        )
        canvas.text(
            self.x_min + self.text_x_offset - 2, self.y_max - 7.0, "--------------->"
        )


# xGMI/PCIe block with wires to fabric
@dataclass
class xGMI_PCIe(RectFrame):
    def draw(self, canvas):
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 1.0, self.y_max - 2.0, self.label)
        canvas.text(self.x_min + 3.0, self.y_max - 5.0, "^   |")
        canvas.text(self.x_min + 3.0, self.y_max - 6.0, "|   |")
        canvas.text(self.x_min + 3.0, self.y_max - 7.0, "|   |")
        canvas.text(self.x_min + 3.0, self.y_max - 8.0, "|   v")


# Fabric Cache Block
@dataclass
class Fabric(RectFrame):
    lat: Dict[str, int] = field(default_factory=dict)

    def draw(self, canvas):
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 6.0, self.y_max - 2.0, "   " + self.label)
        canvas.text(self.x_min + 2.0, self.y_max - 4.0, "Latency (cycles)")
        canvas.rect(self.x_min + 2.0, self.y_max - 9, self.x_max - 2.0, self.y_max - 4.5)

        i = 1
        for k, v in self.lat.items():
            # print(k,v)
            text = "{key:<6}: {val:>6.0f}".format(key=k, val=v)
            canvas.text(self.x_min + 4.0, self.y_max - 4.5 - i, text)
            i = i + 1


# GMI block with wires to fabric
@dataclass
class GMI(RectFrame):
    def draw(self, canvas):
        canvas.text(self.x_min + 3.0, self.y_max + 4.0, "^   |")
        canvas.text(self.x_min + 3.0, self.y_max + 3.0, "|   |")
        canvas.text(self.x_min + 3.0, self.y_max + 2.0, "|   |")
        canvas.text(self.x_min + 3.0, self.y_max + 1.0, "|   v")
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 4.0, self.y_max - 2.0, self.label)


# Wires between fabric and HBM
@dataclass
class Wire_Fabric_HBM(RectFrame):
    text_x_offset: float = 3.0

    rd: int = 0
    wr: int = 0

    def draw(self, canvas):
        canvas.text(
            self.x_min + self.text_x_offset,
            self.y_max,
            "{key:<2}: {val:>4.0f}".format(key="Rd", val=self.rd),
        )
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 1.0, "<-----------")
        canvas.text(
            self.x_min + self.text_x_offset,
            self.y_max - 2.0,
            "{key:<2}: {val:>4.0f}".format(key="Wr", val=self.wr),
        )
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 3.0, "----------->")


# HBM
@dataclass
class HBM(RectFrame):
    def draw(self, canvas):
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 4.0, self.y_max - 2.0, self.label)


# Memory chart pannel for 1 instance
class MemChart:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def draw(self, canvas, normal_unit, metric_dict):
        # ----------------------------------------
        # Overall rect and title
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(
            self.x_min + 2.0, self.y_max - 2.0, "(Normalization: " + normal_unit + ")"
        )

        # Fixme: this is temp solution to filter out non-numeric string
        for k, v in metric_dict.items():
            # print(k, type(v))
            metric_dict[k] = -1 if type(v) == str else v

        # Typically, the drawing order would be: left->right, top->down

        # ----------------------------------------
        # Instr Buff Block
        block_instr_buff = InstrBuff(label="Instr Buff")
        block_instr_buff.x_min = 2.0
        block_instr_buff.x_max = block_instr_buff.x_min + 27.0
        block_instr_buff.y_max = self.y_max - 5.0
        block_instr_buff.y_min = block_instr_buff.y_max - 24.0

        block_instr_buff.wave_occupancy = metric_dict["Wavefront Occupancy"]
        block_instr_buff.wave_life = metric_dict["Wave Life"]

        block_instr_buff.draw(canvas)

        # ----------------------------------------
        # Wires between Instr Buff and Instr Dispatch
        wire_I_I = Wire_InstrBuff_InstrDispatch(
            label="Wire_InstrBuff_InstrDispatch",
            x_min=block_instr_buff.x_max + 1,
            x_max=block_instr_buff.x_max + 7,
            y_min=block_instr_buff.y_min,
            y_max=block_instr_buff.y_max,
        )
        wire_I_I.draw(canvas)

        # ----------------------------------------
        # Instr Dispatch Block
        block_instr_disp = InstrDispatch(label="Instr Dispatch")
        block_instr_disp.x_min = block_instr_buff.x_max + 9.0
        block_instr_disp.x_max = block_instr_disp.x_min + 20.0
        block_instr_disp.y_max = block_instr_buff.y_max
        block_instr_disp.y_min = block_instr_buff.y_min

        block_instr_disp.instrs["SALU"] = metric_dict["SALU"]
        block_instr_disp.instrs["SMEM"] = metric_dict["SMEM"]
        block_instr_disp.instrs["VALU"] = metric_dict["VALU"]
        block_instr_disp.instrs["MFMA"] = metric_dict["MFMA"]
        block_instr_disp.instrs["VMEM"] = metric_dict["VMEM"]
        block_instr_disp.instrs["LDS"] = metric_dict["LDS"]
        block_instr_disp.instrs["GWS"] = metric_dict["GWS"]
        block_instr_disp.instrs["BRANCH"] = metric_dict["BR"]

        block_instr_disp.draw(canvas)

        # ----------------------------------------
        # Exec Block
        block_exec = Exec(label="Exec")
        block_exec.x_min = block_instr_disp.x_max
        block_exec.x_max = block_exec.x_min + 20
        block_exec.y_min = block_instr_disp.y_min - 6
        block_exec.y_max = block_instr_disp.y_max

        block_exec.active_cus = metric_dict["Active CUs"]
        block_exec.num_cus = metric_dict["Num CUs"]
        block_exec.vgprs = metric_dict["VGPR"]
        block_exec.sgprs = metric_dict["SGPR"]
        block_exec.lds_alloc = metric_dict["LDS Allocation"]
        block_exec.scratch_alloc = metric_dict["Scratch Allocation"]
        block_exec.wavefronts = metric_dict["Wavefronts"]
        block_exec.workgroups = metric_dict["Workgroups"]

        block_exec.draw(canvas)

        # ----------------------------------------
        # Wires between Exec block and GDS, LDS, Vector L1 cache
        wires_E_GLV = Wire_E_GLVS(label="Wire_E_GLVS")
        wires_E_GLV.x_min = block_exec.x_max
        wires_E_GLV.x_max = wires_E_GLV.x_min + 16
        wires_E_GLV.y_min = block_instr_disp.y_min
        wires_E_GLV.y_max = block_instr_disp.y_max

        wires_E_GLV.lds_req = metric_dict["LDS Req"]
        wires_E_GLV.vl1_rd = metric_dict["VL1 Rd"]
        wires_E_GLV.vl1_wr = metric_dict["VL1 Wr"]
        wires_E_GLV.vl1_atomic = metric_dict["VL1 Atomic"]
        wires_E_GLV.sl1_rd = metric_dict["VL1D Rd"]

        wires_E_GLV.draw(canvas)

        # ----------------------------------------
        # Wire between Instr Buff and Instr L1 Cache
        wire_InstrBuff_IL1Cache = Wire_InstrBuff_IL1Cache(
            label="Wire_InstrBuff_IL1Cache",
            x_min=block_instr_buff.x_max / 2,
            x_max=block_instr_buff.x_max / 2 + 80,
            y_min=block_exec.y_min - 1,
            y_max=block_instr_buff.y_min,
        )

        wire_InstrBuff_IL1Cache.il1_fetch = metric_dict["IL1 Fetch"]

        wire_InstrBuff_IL1Cache.draw(canvas)

        # ----------------------------------------
        # GDS block
        # block_gds = GDS(label="GDS")
        # block_gds.x_min = wires_E_GLV.x_max + 1
        # block_gds.x_max = block_gds.x_min + 24
        # block_gds.y_max = wires_E_GLV.y_max
        # block_gds.y_min = block_gds.y_max - 5

        # block_gds.gws = metric_dict["gds_gws"]
        # block_gds.latency = metric_dict["gds_latency"]

        # block_gds.draw(canvas)

        # ----------------------------------------
        # LDS block
        block_lds = LDS(label="LDS")
        block_lds.x_min = wires_E_GLV.x_max + 1
        block_lds.x_max = block_lds.x_min + 24
        block_lds.y_max = wires_E_GLV.y_max
        block_lds.y_min = block_lds.y_max - 5

        block_lds.util = metric_dict["LDS Util"]
        block_lds.latency = metric_dict["LDS Latency"]

        block_lds.draw(canvas)

        # ----------------------------------------
        # Vector L1 Cache Block
        block_vector_L1 = VectorL1Cache(label="Vector L1 Cache")
        block_vector_L1.x_min = block_lds.x_min
        block_vector_L1.x_max = block_lds.x_max
        block_vector_L1.y_max = block_lds.y_min - 3
        block_vector_L1.y_min = block_vector_L1.y_max - 9

        block_vector_L1.hit = metric_dict["VL1 Hit"]
        block_vector_L1.latency = metric_dict["VL1 Lat"]
        block_vector_L1.coales = metric_dict["VL1 Coalesce"]
        block_vector_L1.stall = metric_dict["VL1 Stall"]

        block_vector_L1.draw(canvas)

        # ----------------------------------------
        # Scalar L1D Cache block
        block_const_L1 = ScalarL1DCache(label="Scalar L1D Cache")
        block_const_L1.x_min = block_lds.x_min
        block_const_L1.x_max = block_lds.x_max
        block_const_L1.y_max = block_vector_L1.y_min - 3
        block_const_L1.y_min = block_const_L1.y_max - 5

        block_const_L1.hit = metric_dict["VL1D Hit"]
        block_const_L1.latency = metric_dict["VL1D Lat"]

        block_const_L1.draw(canvas)

        # ----------------------------------------
        # Instr L1 Cache Block
        block_instr_L1 = InstrL1Cache(label="Instr L1 Cache")
        block_instr_L1.x_min = block_const_L1.x_min
        block_instr_L1.x_max = block_const_L1.x_max
        block_instr_L1.y_max = block_const_L1.y_min - 3
        block_instr_L1.y_min = block_instr_L1.y_max - 5

        block_instr_L1.hit = metric_dict["IL1 Hit"]
        block_instr_L1.latency = metric_dict["IL1 Lat"]

        block_instr_L1.draw(canvas)

        # ----------------------------------------
        # Wires between Vector L1 cache, Scalar L1D cache, Instr L1 cache and L2 Cache
        wires_L1_L2 = Wires_L1_L2(label="Wires_L1_L2")
        wires_L1_L2.x_min = block_instr_L1.x_max + 4
        wires_L1_L2.x_max = wires_L1_L2.x_min + 14
        wires_L1_L2.y_min = block_instr_L1.y_min
        wires_L1_L2.y_max = block_vector_L1.y_max
        wires_L1_L2.vl1_l2_rd = metric_dict["VL1_L2 Rd"]
        wires_L1_L2.vl1_l2_wr = metric_dict["VL1_L2 Wr"]
        wires_L1_L2.vl1_l2_atomic = metric_dict["VL1_L2 Atomic"]
        wires_L1_L2.sl1_l2_rd = metric_dict["VL1D_L2 Rd"]
        wires_L1_L2.sl1_l2_wr = metric_dict["VL1D_L2 Wr"]
        wires_L1_L2.sl1_l2_atomic = metric_dict["VL1D_L2 Atomic"]
        wires_L1_L2.il1_l2_req = metric_dict["IL1_L2 Rd"]

        wires_L1_L2.draw(canvas)

        # ----------------------------------------
        # L2 Cache Block
        block_L2 = L2Cache(label="L2 Cache")

        block_L2.x_min = wires_L1_L2.x_max + 1
        block_L2.x_max = block_L2.x_min + 24
        block_L2.y_min = block_instr_L1.y_min
        block_L2.y_max = block_lds.y_max

        block_L2.hit = metric_dict["L2 Hit"]
        block_L2.rd = metric_dict["L2 Rd"]
        block_L2.wr = metric_dict["L2 Wr"]
        block_L2.atomic = metric_dict["L2 Atomic"]
        block_L2.rd_lat = metric_dict["L2 Rd Lat"]
        block_L2.wr_lat = metric_dict["L2 Wr Lat"]

        block_L2.draw(canvas)

        # ----------------------------------------
        # Wires between L2 and Fabric
        wires_L2_Fabric = Wire_L2_Fabric(
            label="Wire_L2_Fabric",
            x_min=block_L2.x_max + 1,
            x_max=block_L2.x_max + 16,
            y_min=block_L2.y_max - 18,
            y_max=block_L2.y_max - 10,
        )

        wires_L2_Fabric.rd = metric_dict["Fabric_L2 Rd"]
        wires_L2_Fabric.wr = metric_dict["Fabric_L2 Wr"]
        wires_L2_Fabric.atomic = metric_dict["Fabric_L2 Atomic"]

        wires_L2_Fabric.draw(canvas)

        # ----------------------------------------
        # xGMI/PCIe Block with wires to fabric
        block_xgmi_pcie = xGMI_PCIe(
            label="xGMI/PCIe",
            x_min=wires_L2_Fabric.x_max + 10,
            x_max=wires_L2_Fabric.x_max + 20,
            y_min=block_L2.y_max - 4,
            y_max=block_L2.y_max,
        )
        block_xgmi_pcie.draw(canvas)

        # ----------------------------------------
        # Data Fabric Block
        block_fabric = Fabric(
            label="Fabric",
            x_min=wires_L2_Fabric.x_max + 3,
            x_max=wires_L2_Fabric.x_max + 27,
            y_max=block_xgmi_pcie.y_min - 5,
            y_min=block_xgmi_pcie.y_min - 5 - 11,
        )

        block_fabric.lat["Rd"] = metric_dict["Fabric Rd Lat"]
        block_fabric.lat["Wr"] = metric_dict["Fabric Wr Lat"]
        block_fabric.lat["Atomic"] = metric_dict["Fabric Atomic Lat"]

        block_fabric.draw(canvas)

        # ----------------------------------------
        # GMI Block with wires to fabric
        block_gmi = GMI(
            label="GMI",
            x_min=block_xgmi_pcie.x_min,
            x_max=block_xgmi_pcie.x_max,
            y_min=block_fabric.y_min - 9,
            y_max=block_fabric.y_min - 5,
        )
        block_gmi.draw(canvas)

        # ----------------------------------------
        # Wires between fabric and HBM
        # Wire_Fabric_HBM
        wires_Fabric_HBM = Wire_Fabric_HBM(
            label="Wire_Fabric_HBM",
            x_min=block_fabric.x_max + 1,
            x_max=block_fabric.x_max + 15,
            y_min=block_fabric.y_max - 2,
            y_max=block_fabric.y_max - 4,
        )

        wires_Fabric_HBM.rd = metric_dict["HBM Rd"]
        wires_Fabric_HBM.wr = metric_dict["HBM Wr"]

        wires_Fabric_HBM.draw(canvas)

        # ----------------------------------------
        # HBM Block
        block_hbm = HBM(
            label="HBM",
            x_min=wires_Fabric_HBM.x_max,
            x_max=wires_Fabric_HBM.x_max + 10,
            y_min=block_fabric.y_max - 7,
            y_max=block_fabric.y_max - 3,
        )
        block_hbm.draw(canvas)


def plot_mem_chart(arch, normal_unit, metric_dict):
    """plot memory chart from an arch with given metrics dict"""

    # TODO: verify metrics dict for given arch first

    canvas = Canvas(width=234, height=42, xmax=234, ymax=42)
    mc = MemChart(0, 0, 233, 41)
    mc.draw(canvas, normal_unit, metric_dict)

    # return the plot string stream
    return canvas.plot()


if __name__ == "__main__":
    # Unit test
    metric_dict = {}
    metric_dict["Wavefront Occupancy"] = 1
    metric_dict["Wave Life"] = 2

    metric_dict["SALU"] = 3
    metric_dict["SMEM"] = 4
    metric_dict["VALU"] = 5
    metric_dict["MFMA"] = 6
    metric_dict["VMEM"] = 7
    metric_dict["LDS"] = 8
    metric_dict["GWS"] = 9
    metric_dict["BR"] = 10

    metric_dict["Active CUs"] = 11
    metric_dict["Num CUs"] = 12
    metric_dict["VGPR"] = 13
    metric_dict["SGPR"] = 14
    metric_dict["LDS Allocation"] = 15
    metric_dict["Scratch Allocation"] = 16
    metric_dict["Wavefronts"] = 17
    metric_dict["Workgroups"] = 18

    metric_dict["LDS Req"] = 19
    metric_dict["LDS Util"] = 20
    metric_dict["LDS Latency"] = 21

    metric_dict["VL1 Rd"] = 22
    metric_dict["VL1 Wr"] = 23
    metric_dict["VL1 Atomic"] = 24

    metric_dict["VL1 Hit"] = 25
    metric_dict["VL1 Lat"] = 26
    metric_dict["VL1 Coalesce"] = 27
    metric_dict["VL1 Stall"] = 28

    metric_dict["VL1D Rd"] = 29
    metric_dict["VL1D Hit"] = 30
    metric_dict["VL1D Lat"] = 31

    metric_dict["IL1 Fetch"] = 32
    metric_dict["IL1 Hit"] = 33
    metric_dict["IL1 Lat"] = 34
    metric_dict["IL1_L2 Rd"] = 34

    metric_dict["VL1_L2 Rd"] = 36
    metric_dict["VL1_L2 Wr"] = 37
    metric_dict["VL1_L2 Atomic"] = 38

    metric_dict["VL1D_L2 Rd"] = 39
    metric_dict["VL1D_L2 Wr"] = 40
    metric_dict["VL1D_L2 Atomic"] = 41
    metric_dict["IL1_L2 Rd"] = 42

    metric_dict["L2 Hit"] = 43
    metric_dict["L2 Rd"] = 44
    metric_dict["L2 Wr"] = 45
    metric_dict["L2 Atomic"] = 46
    metric_dict["L2 Rd Lat"] = 47
    metric_dict["L2 Wr Lat"] = 48

    metric_dict["Fabric_L2 Rd"] = 49
    metric_dict["Fabric_L2 Wr"] = 50
    metric_dict["Fabric_L2 Atomic"] = 51

    metric_dict["Fabric Rd Lat"] = 52
    metric_dict["Fabric Wr Lat"] = 53
    metric_dict["Fabric Atomic Lat"] = 54

    metric_dict["HBM Rd"] = 55
    metric_dict["HBM Wr"] = 56

    arch = ""
    normal_unit = "per_kernel"
    print(plot_mem_chart(arch, normal_unit, metric_dict))