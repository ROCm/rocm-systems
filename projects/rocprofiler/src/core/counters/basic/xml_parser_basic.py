#!/usr/bin/python3

from __future__ import print_function
import os, sys, re
import xml.etree.ElementTree as ET
from lxml import etree
import sys

CPP_OUT = "basic_counter.cpp"

if __name__ == "__main__":
    cpp_content = ""
    cpp_content += "/* Copyright (c) 2022 Advanced Micro Devices, Inc.\n"
    cpp_content += "\n"
    cpp_content += (
        " Permission is hereby granted, free of charge, to any person obtaining a copy\n"
    )
    cpp_content += (
        ' of this software and associated documentation files (the "Software"), to deal\n'
    )
    cpp_content += (
        " in the Software without restriction, including without limitation the rights\n"
    )
    cpp_content += (
        " to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
    )
    cpp_content += (
        " copies of the Software, and to permit persons to whom the Software is\n"
    )
    cpp_content += " furnished to do so, subject to the following conditions:\n"
    cpp_content += "\n"
    cpp_content += (
        " The above copyright notice and this permission notice shall be included in\n"
    )
    cpp_content += " all copies or substantial portions of the Software.\n"
    cpp_content += "\n"
    cpp_content += (
        ' THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
    )
    cpp_content += (
        " IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
    )
    cpp_content += (
        " FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
    )
    cpp_content += (
        " AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
    )
    cpp_content += (
        " LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
    )
    cpp_content += (
        " OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n"
    )
    cpp_content += " THE SOFTWARE. */\n"
    cpp_content += "\n"
    cpp_content += "#include <cassert>\n"
    cpp_content += '#include "src/utils/helper.h"\n'
    cpp_content += "\n"
    cpp_content += '#include "src/core/counters/basic/basic_counter.h"\n'
    cpp_content += '#include "src/core/hardware/hsa_info.h"\n'
    cpp_content += "\n"
    cpp_content += "#define ASSERTM(exp, msg) assert(((void)msg, exp))\n"
    cpp_content += "\n"
    cpp_content += "#pragma GCC diagnostic push\n"
    cpp_content += '#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"\n'
    cpp_content += "namespace Counter {\n"
    cpp_content += "\n"
    cpp_content += "BasicCounter::BasicCounter(uint64_t event_id, std::string block_id,\n"
    cpp_content += (
        "                           std::string name, std::string description,\n"
    )
    cpp_content += "                           std::string gpu_name)\n"
    cpp_content += "    : Counter(name, description, gpu_name),\n"
    cpp_content += "      event_id_(event_id),\n"
    cpp_content += "      block_id_(block_id) {\n"
    cpp_content += "  AddCounterToCounterMap();\n"
    cpp_content += "}\n"
    cpp_content += "\n"
    cpp_content += "BasicCounter::~BasicCounter() {}\n"
    cpp_content += "\n"
    cpp_content += "uint64_t BasicCounter::GetBasicCounterID() {\n"
    cpp_content += "  return GetCounterID();\n"
    cpp_content += "}\n"
    cpp_content += "\n"
    cpp_content += "uint64_t BasicCounter::GetEventId() { return event_id_; }\n"
    cpp_content += "std::string BasicCounter::GetBlockId() { return block_id_; }\n"
    cpp_content += "std::string BasicCounter::GetName() { return Counter::GetName(); }\n"
    cpp_content += "\n"
    cpp_content += (
        "bool BasicCounter::GetValue(uint64_t* value, int64_t instance_id = -1) {\n"
    )
    cpp_content += "  Agent::CounterHardwareInfo* agent_info =\n"
    cpp_content += (
        "      reinterpret_cast<Agent::CounterHardwareInfo*>(counter_hw_info);\n"
    )
    cpp_content += "  if ((agent_info->getNumInstances() > 1 && instance_id == -1) ||\n"
    cpp_content += (
        "      instance_id < -1 || instance_id >= agent_info->getNumInstances())\n"
    )
    cpp_content += "    return false;\n"
    cpp_content += "  if (instance_id == -1) *value = instances_values_[0];\n"
    cpp_content += "  *value = instances_values_[instance_id];\n"
    cpp_content += "  return true;\n"
    cpp_content += "}\n"
    cpp_content += "\n"
    cpp_content += "uint64_t BasicCounter::GetValue(int64_t instance_id) {\n"
    cpp_content += "  Agent::CounterHardwareInfo* agent_info =\n"
    cpp_content += (
        "      reinterpret_cast<Agent::CounterHardwareInfo*>(counter_hw_info);\n"
    )
    cpp_content += "  if ((agent_info->getNumInstances() > 1 && instance_id == -1) ||\n"
    cpp_content += (
        "      instance_id < -1 || instance_id >= agent_info->getNumInstances())\n"
    )
    cpp_content += '    throw(std::string("Error: Wrong number of instances (") +\n'
    cpp_content += "                std::to_string(agent_info->getNumInstances()) +\n"
    cpp_content += '                ") OR Instance ID is less than 0 ");\n'
    cpp_content += "  if (instance_id == -1) return instances_values_[0];\n"
    cpp_content += "  return instances_values_[instance_id];\n"
    cpp_content += "}\n"
    cpp_content += "\n"
    cpp_content += "uint64_t BasicCounter::avr(int64_t instances_count) {\n"
    cpp_content += "  Agent::CounterHardwareInfo* agent_info =\n"
    cpp_content += (
        "      reinterpret_cast<Agent::CounterHardwareInfo*>(counter_hw_info);\n"
    )
    cpp_content += "  if (agent_info->getNumInstances() > instances_count)\n"
    cpp_content += '    throw(std::string("Error: Number of instances (") +\n'
    cpp_content += "                std::to_string(agent_info->getNumInstances()) +\n"
    cpp_content += '                ") is greater than the given instance count(" +\n'
    cpp_content += '                std::to_string(instances_count) + ")");\n'
    cpp_content += "  uint64_t result = 0;\n"
    cpp_content += "  int64_t instance_id;\n"
    cpp_content += (
        "  for (instance_id = 0; instance_id < instances_count; instance_id++) {\n"
    )
    cpp_content += "    uint64_t value;\n"
    cpp_content += "    if (GetValue(&value, instance_id)) result += value;\n"
    cpp_content += "  }\n"
    cpp_content += "  return result / instances_count;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t BasicCounter::max(int64_t instances_count) {\n"
    cpp_content += "  uint64_t result = 0;\n"
    cpp_content += "  int64_t instance_id;\n"
    cpp_content += (
        "  for (instance_id = 0; instance_id < instances_count; instance_id++) {\n"
    )
    cpp_content += "    uint64_t value;\n"
    cpp_content += (
        "    if (GetValue(&value, instance_id) && result < value) result = value;\n"
    )
    cpp_content += "  }\n"
    cpp_content += "  return result;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t BasicCounter::min(int64_t instances_count) {\n"
    cpp_content += "  int64_t instance_id;\n"
    cpp_content += "  uint64_t result = 0;\n"
    cpp_content += (
        "  for (instance_id = 0; instance_id < instances_count; instance_id++) {\n"
    )
    cpp_content += "    uint64_t value;\n"
    cpp_content += (
        "    if (GetValue(&value, instance_id) && result > value) result = value;\n"
    )
    cpp_content += "  }\n"
    cpp_content += "  return result;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t BasicCounter::sum(int64_t instances_count) {\n"
    cpp_content += "  int64_t instance_id;\n"
    cpp_content += "  uint64_t result = 0;\n"
    cpp_content += (
        "  for (instance_id = 0; instance_id < instances_count; instance_id++) {\n"
    )
    cpp_content += "    uint64_t value;\n"
    cpp_content += "    if (GetValue(&value, instance_id)) result += value;\n"
    cpp_content += "  }\n"
    cpp_content += "  return result;\n"
    cpp_content += "}\n"
    cpp_content += "\n"
    cpp_content += "uint64_t operator+(BasicCounter counter, const uint64_t number) {\n"
    cpp_content += "  [[maybe_unused]] uint64_t value = 0;\n"
    cpp_content += (
        '  ASSERTM(counter.GetValue(&value), "Error: Counter has no value!");\n'
    )
    cpp_content += "  return number + value;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t operator-(BasicCounter counter, const uint64_t number) {\n"
    cpp_content += "  [[maybe_unused]] uint64_t value = 0;\n"
    cpp_content += (
        '  ASSERTM(counter.GetValue(&value), "Error: Counter has no value!");\n'
    )
    cpp_content += "  return number - value;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t operator*(BasicCounter counter, const uint64_t number) {\n"
    cpp_content += "  [[maybe_unused]] uint64_t value = 0;\n"
    cpp_content += (
        '  ASSERTM(counter.GetValue(&value), "Error: Counter has no value!");\n'
    )
    cpp_content += "  return number * value;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t operator/(BasicCounter counter, const uint64_t number) {\n"
    cpp_content += "  [[maybe_unused]] uint64_t value = 0;\n"
    cpp_content += (
        '  ASSERTM(counter.GetValue(&value), "Error: Counter has no value!");\n'
    )
    cpp_content += "  return number / value;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t operator^(BasicCounter counter, const uint64_t number) {\n"
    cpp_content += "  [[maybe_unused]] uint64_t value = 0;\n"
    cpp_content += (
        '  ASSERTM(counter.GetValue(&value), "Error: Counter has no value!");\n'
    )
    cpp_content += "  return number ^ value;\n"
    cpp_content += "}\n"
    cpp_content += "\n"
    cpp_content += "uint64_t operator+(BasicCounter counter1, BasicCounter counter2) {\n"
    cpp_content += "  [[maybe_unused]] uint64_t value1 = 0;\n"
    cpp_content += (
        '  ASSERTM(counter1.GetValue(&value1), "Error: Counter has no value!");\n'
    )
    cpp_content += "  [[maybe_unused]] uint64_t value2 = 0;\n"
    cpp_content += (
        '  ASSERTM(counter2.GetValue(&value2), "Error: Counter has no value!");\n'
    )
    cpp_content += "  return value1 + value2;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t operator-(BasicCounter counter1, BasicCounter counter2) {\n"
    cpp_content += "  [[maybe_unused]] uint64_t value1 = 0;\n"
    cpp_content += (
        '  ASSERTM(counter1.GetValue(&value1), "Error: Counter has no value!");\n'
    )
    cpp_content += "  [[maybe_unused]] uint64_t value2 = 0;\n"
    cpp_content += (
        '  ASSERTM(counter2.GetValue(&value2), "Error: Counter has no value!");\n'
    )
    cpp_content += "  return value1 - value2;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t operator*(BasicCounter counter1, BasicCounter counter2) {\n"
    cpp_content += "  [[maybe_unused]] uint64_t value1 = 0;\n"
    cpp_content += (
        '  ASSERTM(counter1.GetValue(&value1), "Error: Counter has no value!");\n'
    )
    cpp_content += "  [[maybe_unused]] uint64_t value2 = 0;\n"
    cpp_content += (
        '  ASSERTM(counter2.GetValue(&value2), "Error: Counter has no value!");\n'
    )
    cpp_content += "  return value1 * value2;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t operator/(BasicCounter counter1, BasicCounter counter2) {\n"
    cpp_content += "  [[maybe_unused]] uint64_t value1 = 0;\n"
    cpp_content += (
        '  ASSERTM(counter1.GetValue( & value1), "Error: Counter has no value!");\n'
    )
    cpp_content += "  [[maybe_unused]] uint64_t value2 = 0;\n"
    cpp_content += (
        '  ASSERTM(counter2.GetValue( & value2), "Error: Counter has no value!");\n'
    )
    cpp_content += " return value1 / value2;\n"
    cpp_content += "}\n"
    cpp_content += "uint64_t operator^(BasicCounter counter1, BasicCounter counter2) {\n"
    cpp_content += "  [[maybe_unused]] uint64_t value1 = 0;\n"
    cpp_content += (
        '  ASSERTM(counter1.GetValue(&value1), "Error: Counter has no value!");\n'
    )
    cpp_content += "  [[maybe_unused]] uint64_t value2 = 0;\n"
    cpp_content += (
        '  ASSERTM(counter2.GetValue(&value2), "Error: Counter has no value!");\n'
    )
    cpp_content += "  return value1 ^ value2;\n"
    cpp_content += "}\n"
    cpp_content += "\n"
    cpp_content += "static std::map<uint64_t, BasicCounter> basic_counters;\n"
    cpp_content += "\n"
    cpp_content += "BasicCounter* GetGeneratedBasicCounter(uint64_t id) {\n"
    cpp_content += "  return &basic_counters.at(id);\n"
    cpp_content += "}\n"
    cpp_content += "\n"
    cpp_content += "void ClearBasicCounters() {\n"
    cpp_content += "  basic_counters.clear();\n"
    cpp_content += "}\n"
    cpp_content += "\n"
    cpp_content += "/**\n"
    cpp_content += " * @brief Basic Counters\n"
    cpp_content += " *\n"
    cpp_content += " * @{\n"
    cpp_content += " */\n"
    cpp_content += "uint64_t GetBasicCounter(const char* name, const char* gpu_name) {\n"
    cpp_content += "  std::string gpu;\n"
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    xml_file = ET.parse(sys.argv[1] + "/gfx_metrics.xml", parser=parser)
    root = xml_file.getroot()
    for gpu in root:
        cpp_content += (
            "\n\t/**\n\t * @brief Basic " + gpu.tag + " counters\n\t *\n\t * @{\n\t */\n"
        )
        cpp_content += '\tgpu = "' + gpu.tag + '";\n\n'
        cpp_content += "\tif (strncmp(gpu_name, gpu.c_str(), gpu.length())==0) {\n"
        for child in gpu:
            cpp_content += (
                "\t/**\n\t * Basic Counter: "
                + child.attrib["name"]
                + "\n\t *\n\t * "
                + child.attrib["descr"]
                + '\n\t */\n\tif (strcmp(name, "'
                + child.attrib["name"]
                + '")==0) {\n\t\tbasic_counters.emplace('
                + child.attrib["event"]
                + ", BasicCounter{"
                + child.attrib["event"]
                + ', "'
                + child.attrib["block"]
                + '", "'
                + child.attrib["name"]
                + '", "'
                + child.attrib["descr"]
                + '", "'
                + gpu.tag
                + '"});\n\t\treturn '
                + child.attrib["event"]
                + ";\n\t}\n"
            )
        cpp_content += "\t}\n\n\t/**\n\t * @}\n\t */\n"
    cpp_content += (
        '  throw("Couldn\'t find the required Counter name for the mentioned GPU!");\n'
    )
    cpp_content += "  return 0;\n"
    cpp_content += "}\n"
    cpp_content += "/**\n"
    cpp_content += " * @}\n"
    cpp_content += " */\n"
    cpp_content += "\n"
    cpp_content += "}  // namespace Counter\n"
    cpp_content += "\n"
    cpp_content += "#pragma GCC diagnostic pop\n"
    print('Generating "' + sys.argv[2] + '"')
    f = open(sys.argv[2], "w")
    f.write(cpp_content[:-1])
    f.close()
