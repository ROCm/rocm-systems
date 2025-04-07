#!/usr/bin/python3

from collections import deque
from re import sub
import xml.etree.ElementTree as ET
from lxml import etree
import ast
import sys

ops = {"Div": "/", "Mult": "*", "Add": "+", "Sub": "-"}
calls = {"avr", "max", "min", "sum"}


def parse_expr(gpu_tag, data):
    global exprs_counters
    global exprs_counters_init
    global expr_print
    global counter_count
    global counters_dictionary
    expr_queue = deque()
    for line in data.split("\n"):
        if "Constant" in line:
            number = line.split("(")[1].split(")")[0]
            expr_queue.append("(uint64_t)" + number)
        if "Name" in line:
            name = line.split("'")[1]
            if name in calls:
                expr_queue.append(name)
            else:
                if not name in exprs_counters:
                    exprs_counters += "getGeneratedBasicCounter(" + name + "_id), "
                    exprs_counters_init += (
                        "\n\t\tuint64_t "
                        + name
                        + '_id = getBasicCounter("'
                        + name
                        + '", "'
                        + gpu_tag
                        + '");'
                    )
                    counters_dictionary[name] = counter_count
                    counter_count += 1
                expr_queue.append(
                    "counter.getBasicCounterFromDerived("
                    + str(counters_dictionary[name])
                    + ")"
                )
        op = line.split("(")[0]
        if op in ops:
            expr_queue.append(ops[op])
    expr_print += "\n\t\t\t\treturn "
    i = 0
    for element in expr_queue:
        if element in calls:
            i = 1
            call = element
        elif i == 1:
            expr_print += element + "." + call + "("
            call = ""
            i = 2
        elif i == 2:
            expr_print += element + ")"
            i = 0
        else:
            expr_print += element
            if "counter.getBasicCounterFromDerived" == element[0:34]:
                expr_print += "->getValue()"


if __name__ == "__main__":
    global exprs_counters
    global exprs_counters_init
    global expr_print
    global counter_count
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    xml_file = ET.parse(sys.argv[1] + "/metrics.xml", parser=parser)
    root = xml_file.getroot()
    print("uint64_t getDerivedCounter(const char* name, const char* gpu_name) {")
    for gpu in root:
        print(
            "\n\t/**\n\t * @brief Derived " + gpu.tag + " counters\n\t *\n\t * @{\n\t */"
        )
        print('\tif (strcmp(gpu_name, "' + gpu.tag + '")==0) {')
        for child in gpu:
            exprs_counters = ""
            exprs_counters_init = ""
            expr_print = ""
            counter_count = 0
            counters_dictionary = {}
            parse_expr(
                gpu.tag.split("_")[0],
                ast.dump(
                    ast.parse(child.attrib["expr"], mode="eval"),
                    annotate_fields=False,
                    include_attributes=False,
                    indent=0,
                ),
            )
            print(
                "\t/**\n\t * Derived Counter: "
                + child.attrib["name"]
                + "\n\t *\n\t * "
                + child.attrib["descr"]
                + '\n\t */\n\tif (strcmp(name, "'
                + child.attrib["name"]
                + '")==0) {'
                + exprs_counters_init
                + '\n\t\tDerivedCounter counter = DerivedCounter("'
                + child.attrib["name"]
                + '", "'
                + child.attrib["descr"]
                + '", "'
                + gpu.tag.split("_")[0]
                + '");'
            )
            exprs_counter_count = 0
            for expr_counter in exprs_counters[0:-2].split(", "):
                print(
                    "\n\t\tcounter.addBasicCounter("
                    + str(exprs_counter_count)
                    + ", "
                    + expr_counter
                    + ");"
                )
                exprs_counter_count += 1
            # print("\n\t\tcounter.evaluate_metric = [counter]() {" + expr_print + ";\n\t\t\t};")
            print(
                "\n\t\tderived_counters.emplace(counter.getMetricId(), counter);\n\t\treturn counter.getMetricId();\n\t}"
            )
        print("\t}\n\n\t/**\n\t * @}\n\t */")
    print("\n\treturn 0;\n}\n")
