import os
import re
import json
import csv
from collections import defaultdict

if __name__ == "__main__":
    test_case_pattern = re.compile(r'TEST_CASE\("([^"]+)"\)')
    template_test_case_pattern = re.compile(r'TEMPLATE_TEST_CASE\("([^"]+)",\s*((?:[^;]*?))\) ', re.DOTALL)
    test_cases = set()
    disabled_tags = defaultdict(list)
    categories = defaultdict(str)

    for root, _, files in os.walk(os.getcwd()):
        parts = os.path.relpath(root, os.getcwd()).split(os.sep)
        category = os.path.join(*parts[1:]) if len(parts) > 1 else ""
        for file in files:
            if file.endswith('.cpp') or file.endswith('.h') or file.endswith('.hh') or file.endswith('.cc'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='windows-1251') as f:
                    content = f.read()
                    for match in test_case_pattern.finditer(content):
                        test_cases.add(match.group(1))
                        categories[match.group(1)] = category
                    for match in template_test_case_pattern.finditer(content):
                        name = match.group(1)
                        arguments = match.group(2).replace('\n', ' ').strip()
                        if "TestParams<" in arguments:
                            for argument in arguments.split("TestParams<")[1:]:
                                argument = argument.split(">)")[0].replace('"', '').strip()
                                if argument != "":
                                    test_cases.add(f"{name} - TestParams<{argument}>")
                                    categories[f"{name} - TestParams<{argument}>"] = category
                        else:
                            for argument in arguments.split(",")[1:]:
                                argument = argument.strip()
                                if argument != "":
                                    test_cases.add(f"{name} - {argument}")
                                    categories[f"{name} - {argument}"] = category

    print(f"Test cases found: {len(test_cases)}")

    config_dir = os.path.join(os.getcwd(), "catch/hipTestMain/config")
    current_tags = []
    for config_name in os.listdir(config_dir):
        config_content = open(os.path.join(config_dir, config_name), encoding='windows-1251').read()
        base_tag = config_name.replace("config_", "")
        current_tag = "" + base_tag
        if config_name.endswith(".json"):
            disabled_list = json.loads(config_content)["DisabledTests"]
            current_tag = current_tag.replace(".json", "")
            for test in disabled_list:
                if test in test_cases:
                    disabled_tags[current_tag].append(test)
        else:
            for line in config_content.split("\n"):
                if line.strip().startswith("#if defined"):
                    arr = [word.replace("||", "").strip() for word in line.strip()[3:].split("defined") if word.strip() != ""]
                    current_tags = []
                    for tag in arr:
                        if tag != "COMMON":
                            current_tags.append(tag)
                    continue
                if line.replace(',', '').replace('"', '').strip() in test_cases:
                    test_name = line.replace(',', '').replace('"', '').strip()
                    disabled_tags[base_tag].append(test_name)
                    for current_tag in current_tags:
                        disabled_tags[current_tag].append(test_name)


    headers = ["Test name", "Category"]
    headers.extend(disabled_tags.keys())
    with open("result.csv", "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        rows = []
        for test in test_cases:
            row = {"Test name": test, "Category": categories[test]}
            for key in disabled_tags.keys():
                row[key] = "disabled" if test in disabled_tags[key] else "enabled"
            rows.append(row)
        writer.writerows(rows)

