/*
Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef MODEL_DESCS_H_
#define MODEL_DESCS_H_

typedef struct NodeModelDesc {
    const char *filename;
    const char *description;
} NodeModelDesc;

inline NodeModelDesc model_descs[] = {
  // GFX 906
  {"topo_4p1h.xml",                      " 4gfx906 1H2XGMI  1NIC 1Intel A"},
  {"topo_4p1h_1.xml",                    " 4gfx906 1H2XGMI  2NIC 2Intel A"},
  {"topo_8p_rome.xml",                   " 8gfx906 2H2XGMI  1NIC 2AMD   A"},
  {"topo_8p_rome_n2.xml",                " 8gfx906 2H2XGMI  1NIC 4AMD   A"},
  {"topo_8p_rome_n4.xml",                " 8gfx906 2H2XGMI  1NIC 7AMD   A"},
  {"topo_4p2h.xml",                      " 8gfx906 2H2XGMI  1NIC 1Intel A"},
  {"topo_4p2h_1.xml",                    " 8gfx906 2H2XGMI  1NIC 1Intel B"},
  {"topo_4p2h_2nic.xml",                 " 8gfx906 2H2XGMI  2NIC 1Intel A"},
  {"topo_8p_rome_n2_1.xml",              " 8gfx906 2H2XGMI  2NIC 4AMD   A"},
  {"topo_8p_rome_n2_2.xml",              " 8gfx906 2H2XGMI  2NIC 4AMD   B"},
  {"topo_8p_ts1.xml",                    " 8gfx906 2H2XGMI  2NIC 4AMD   C"},
  {"topo_8p_ts1_1.xml",                  " 8gfx906 2H2XGMI  2NIC 4AMD   D"},
  {"topo_8p_ts1_n4.xml",                 " 8gfx906 2H2XGMI  2NIC 8AMD   A"},
  {"topo_8p_ts1_n4_1.xml",               " 8gfx906 2H2XGMI  2NIC 8AMD   B"},
  {"topo_8p_ts1_n4_2.xml",               " 8gfx906 2H2XGMI  3NIC 8AMD   C"},
  {"topo_8p_pcie.xml",                   " 8gfx906 PCIe     1NIC 1Intel A"},
  {"topo_8p_pcie_1.xml",                 " 8gfx906 PCIe     1NIC 1Intel B"},
  {"topo_8p_pcie_2nic.xml",              " 8gfx906 PCIe     2NIC 1Intel A"},
  {"topo_8p_rome_pcie.xml",              " 8gfx906 PCIe     2NIC 2AMD2  A"},
  // GFX 908
  {"topo_4p3l.xml",                      " 4gfx908 1H3XGMI  2NIC 1Intel A"},
  {"topo_8p6l.xml",                      " 8gfx908 1H6XGMI  1NIC 2AMD   A"},
  {"topo_8p6l_1nic.xml",                 " 8gfx908 1H6XGMI  1NIC 2AMD   B"},
  {"topo_8p6l_2nic.xml",                 " 8gfx908 1H6XGMI  2NIC 2AMD   A"},
  {"topo_8p6l_3nic.xml",                 " 8gfx908 1H6XGMI  3NIC 2AMD   A"},
  {"topo_8p6l_4nic.xml",                 " 8gfx908 1H6XGMI  4NIC 2AMD   A"},
  {"topo_8p6l_5nic.xml",                 " 8gfx908 1H6XGMI  5NIC 2AMD   A"},
  {"topo_8p6l_6nic.xml",                 " 8gfx908 1H6XGMI  6NIC 2AMD   A"},
  {"topo_4p3l_ia.xml",                   " 8gfx908 2H3XGMI  1NIC 1Intel A"},
  {"topo_4p3l_2h.xml",                   " 8gfx908 2H3XGMI  1NIC 4AMD   A"},
  {"topo_4p3l_n2.xml",                   " 8gfx908 2H3XGMI  1NIC 4AMD   B"},
  {"topo_4p3l_n2_1.xml",                 " 8gfx908 2H3XGMI  1NIC 4AMD   C"},
  {"topo_collnet_n1.xml",                " 8gfx908 2H3XGMI  1NIC 4AMD   D"},
  {"topo_8p_rome_vm1.xml",               " 8gfx908 2H3XGMI  1NIC 4AMD   E"},
  {"topo_4p3l_n4.xml",                   " 8gfx908 2H3XGMI  1NIC 7AMD   A"},
  {"topo_8p_rome_n4_1.xml",              " 8gfx908 2H3XGMI  1NIC 7AMD   B"},
  {"topo_8p_rome_4nics.xml",             " 8gfx908 2H3XGMI  4NIC 4AMD   A"},
  {"topo_collnet_n4.xml",                " 8gfx908 2H3XGMI  4NIC 4AMD   B"},
  {"topo_8p_rome_4n_1.xml",              " 8gfx908 2H3XGMI  4NIC 4AMD   C"},
  {"topo_8p_rome_4n_2.xml",              " 8gfx908 2H3XGMI  4NIC 4AMD   D"},
  {"topo_8p_4nics.xml",                  " 8gfx908 2H3XGMI  4NIC 4AMD   E"},
  {"topo_4p4h.xml",                      "16gfx908 2H3XGMI 16NIC 1AMD   A"},
  // GFX 910
  {"topo_3p_pcie.xml",                   " 3gfx910 PCIe     1NIC 2AMD   A"},
  {"topo_3p_pcie_1.xml",                 " 3gfx910 PCIe     1NIC 2AMD   B"},
  {"topo_8p_90a.xml",                    " 8gfx910 2H3XGMI  1NIC 1AMD   A"},
  {"topo_8p_90a_1.xml",                  " 8gfx910 2H3XGMI  1NIC 3AMD   A"},
  {"topo_8p1h_2.xml",                    " 8gfx910 2H3XGMI  2NIC 4AMD   A"},
  {"topo_8p1h.xml",                      " 8gfx910 2H3XGMI  4NIC 2AMD   A"},
  {"topo_8p1h_n1.xml",                   " 8gfx910 2H3XGMI  4NIC 2AMD   B"},
  {"topo_8p1h_1.xml",                    " 8gfx910 2H3XGMI  4NIC 2AMD   C"},
  {"topo_8p1h_3.xml",                    " 8gfx910 2H3XGMI  4NIC 4AMD   A"},
  {"topo_8p1h_4.xml",                    " 8gfx910 2H3XGMI  8NIC 2AMD   A"},
  {"topo_8p1h_5.xml",                    " 8gfx910 2H3XGMI  8NIC 2AMD   B"},
  {"topo_16p1h.xml",                     "16gfx910 2H3XGMI  8NIC 4AMD   A"},
  {"topo_16p1h_vm.xml",                  "16gfx910 2H3XGMI  8NIC 4AMD   B"},
  // GFX 942
  {"topo_4p_942.xml",                    " 4gfx942 1H3XGMI  4NIC 4AMD2  A"},
  {"topo_8p_942.xml",                    " 8gfx942 1H7XGMI  8NIC 2Intel A"},
  {"topo_8p_942vm.xml",                  " 8gfx942 1H7XGMI  8NIC 2Intel B"},
  {"topo_16p_gio-1s-1rp-cascade.xml",    "16gfx942 2H7XGMI  1NIC 2AMD   A"},
  {"topo_16p_gio-3s-1rp-split-flat.xml", "16gfx942 2H7XGMI  1NIC 2AMD   B"},
  // GFX 950
  {"topo_8p_950.xml",                    " 8gfx950 1H7XGMI  8NIC 2AMD   A"},
};

inline const int num_models = sizeof(model_descs) / sizeof(*model_descs);

#endif // MODEL_DESCS_H_
