/*
Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef MODEL_H_
#define MODEL_H_

#include <vector>
#include "topo.h"
#include "xml.h"
#include "utils.h"

class NodeModel {
private:

public:
  std::vector<struct ncclTopoSystem*> systems;
  uint64_t hostHash;  // auto-generated
  uint64_t pidHash;   // auto-generated
  int nodeId;
  int firstRank;
  int currRank;

  NodeModel(const char *xml_file) {
    char filename[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", filename, PATH_MAX);
    while (--count > 0) {
      if (filename[count] == '/') {
        filename[count+1] = 0;
        break;
      }
    };
    strcat(filename, "models/");
    strcat(filename, xml_file);
    struct ncclTopoSystem* system;
    ncclTopoGetSystem(filename, &system);
    systems.push_back(system);
    for (int i=0; i<getNumGpus()-1; i++) {
      ncclTopoGetSystem(filename, &system);
      systems.push_back(system);
    }
    hostHash = ((uint64_t)rand() << 32) | rand();
    pidHash = ((uint64_t)rand() << 32) | rand();
  }

  struct ncclTopoSystem* getSystem(int rank) { return systems[rank-firstRank]; }

  int getNumGpus() {
    return systems[0]->nodes[GPU].count;
  }

  int rankToCudaDev(int rank) {
    for (int i=0; i<getNumGpus(); i++) {
      if (rank == systems[0]->nodes[GPU].nodes[i].gpu.rank)
        return systems[0]->nodes[GPU].nodes[i].gpu.dev;
    }
    return -1;
  }

  int64_t getGpuBusId(int rank) {
    for (int i=0; i<getNumGpus(); i++) {
      if (rank == systems[0]->nodes[GPU].nodes[i].gpu.rank)
        return systems[0]->nodes[GPU].nodes[i].id;
    }
    return -1;
  }

  int busIdToCudaDev(int64_t busId) {
    for (int i=0; i<getNumGpus(); i++)
      if (systems[0]->nodes[GPU].nodes[i].id == busId)
        return systems[0]->nodes[GPU].nodes[i].gpu.dev;
    return -1;
  }

  void setRanks() {
    for (int r=0; r<getNumGpus(); r++)
      for (int i=0; i<getNumGpus(); i++)
        systems[r]->nodes[GPU].nodes[i].gpu.rank += firstRank;
  }

  int p2pCanConnect(int device1, int device2) { return 1; }
  int shmCanConnect(int device1, int device2) { return 1; }
  int netCanConnect(int device1, int device2) { return 1; }

  ~NodeModel() {}
};

class NetworkModel {
private:
  int nRanks;
  std::vector<NodeModel*> nodes;

public:
  void AddNode(NodeModel* node) {
    node->nodeId = nodes.size();
    node->firstRank = nRanks;
    node->setRanks();
    nRanks += node->getNumGpus();
    nodes.push_back(node);
  }

  NodeModel* GetNode(int rank) {
    for (auto & node : nodes) {
      if (rank >= node->firstRank && rank < node->firstRank+node->getNumGpus()) {
        node->currRank = rank;
        return node;
      }
    }
    return NULL;
  }

  int GetNNodes() { return nodes.size(); }
  int GetNRanks() { return nRanks; }

  NetworkModel() : nRanks(0) {}
};

#endif