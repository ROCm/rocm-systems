#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import re
import os
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata


class GoApiRefDirective(Directive):
    """
    Directive for generating Go API reference documentation.

    Usage:
    .. go-api-ref:: path/to/gofile.go
       :section: gpu
    """

    required_arguments = 1  # Requires one argument: the path to the Go file
    optional_arguments = 0
    has_content = False
    option_spec = {
        "section": directives.unchanged,  # Optional section filter
    }

    def run(self):
        # Get the path to the Go file
        go_file_path = self.arguments[0]
        env = self.state.document.settings.env

        # Get the section filter if provided
        section_filter = self.options.get("section", None)

        # Resolve the path relative to the document
        doc_dir = Path(env.doc2path(env.docname)).parent
        source_path = (doc_dir / go_file_path).resolve()

        # Check if the file exists
        if not source_path.exists():
            msg = f"Go source file not found: {source_path}"
            return [nodes.warning("", nodes.paragraph("", msg))]

        # Parse the Go file and generate documentation
        functions = parse_go_file(str(source_path))

        # Create a container for the API documentation
        container = nodes.container()
        container["classes"].append("go-api-reference")

        # Add the API documentation to the container
        content = generate_rst_content(functions, section_filter)
        self.state_machine.insert_input(content, source=str(source_path))

        return [container]


def parse_go_file(file_path):
    """Parse a Go file and extract function documentation."""
    with open(file_path, "r") as f:
        content = f.read()

    # Pattern to match function documentation and definition
    pattern = r"(\/\/[^\n]*(?:\n\/\/[^\n]*)*)\n\s*func\s+([A-Za-z0-9_]+)\s*\((.*?)\)\s*(\(.*?\)|\w+)\s*\{"
    matches = re.findall(pattern, content, re.DOTALL)

    functions = []
    for match in matches:
        doc_comment = match[0]
        func_name = match[1]
        params = match[2].strip()
        return_type = match[3].strip()

        # Process the comment lines
        doc_lines = []
        for line in doc_comment.split("\n"):
            if line.strip().startswith("//"):
                # Remove the comment marker and one space after it (if present)
                comment_text = line.strip()[2:]
                if comment_text.startswith(" "):
                    comment_text = comment_text[1:]
                doc_lines.append(comment_text)

        # Extract sections from the doc comment
        description = []
        input_params = []
        output_params = []
        example = []

        current_section = "description"

        for line in doc_lines:
            if line.startswith("Input parameter"):
                current_section = "input"
                input_params.append(line)
            elif line.startswith("Output:"):
                current_section = "output"
                output_params.append(line)
            elif line.startswith("Example:"):
                current_section = "example"
                example.append(line)
            elif current_section == "description":
                description.append(line)
            elif current_section == "input":
                input_params.append(line)
            elif current_section == "output":
                output_params.append(line)
            elif current_section == "example":
                example.append(line)

        # Combine description lines into a single line
        desc_text = " ".join([line.strip() for line in description if line.strip()])

        # Combine output lines into a single line
        output_text = " ".join([line.strip() for line in output_params if line.strip()])

        # Determine the section based on function name
        parts = func_name.split("_")
        section = parts[1] if len(parts) > 1 else "other"

        functions.append(
            {
                "name": func_name,
                "params": params,
                "return_type": return_type,
                "description": desc_text,
                "input_params": "\n".join(input_params).strip(),
                "output_params": output_text,
                "example": "\n".join(example).strip(),
                "section": section.lower(),  # Store the section for filtering
            }
        )

    return functions


def generate_rst_content(functions, section_filter=None):
    """Generate reStructuredText content from parsed functions."""
    lines = []

    # Filter functions by section if a filter is provided
    if section_filter:
        section_filter = section_filter.lower()
        functions = [f for f in functions if f["section"] == section_filter]

        if not functions:
            lines.append(f"No functions found in section: {section_filter}")
            return lines

    # Group functions by prefix if no section filter is provided
    if not section_filter:
        # Group functions by prefix (e.g., GO_gpu_, GO_cpu_)
        function_groups = {}
        for func in functions:
            section = func["section"]
            if section not in function_groups:
                function_groups[section] = []
            function_groups[section].append(func)

        # Define the order of sections (GPU first, then CPU, then others)
        section_order = []

        # Add GPU section first if it exists
        if "gpu" in function_groups:
            section_order.append("gpu")

        # Add CPU section next if it exists
        if "cpu" in function_groups:
            section_order.append("cpu")

        # Add all other sections in alphabetical order
        for prefix in sorted(function_groups.keys()):
            if prefix not in ["gpu", "cpu"]:
                section_order.append(prefix)

        # Write each group in the specified order
        for section in section_order:
            funcs = function_groups[section]
            lines.append(f"{section.upper()} Functions")
            lines.append("-" * len(f"{section.upper()} Functions"))
            lines.append("")

            for func in funcs:
                add_function_documentation(lines, func)
    else:
        # If a section filter is provided, just document those functions without section headers
        for func in functions:
            add_function_documentation(lines, func)

    return lines


def add_function_documentation(lines, func):
    """Add documentation for a single function to the lines list."""
    lines.append(func['name'])
    lines.append("~" * len(f"``{func['name']}``"))
    lines.append("")

    # Function signature
    return_type = func["return_type"]
    if return_type.startswith("(") and return_type.endswith(")"):
        return_type = return_type[1:-1]

    lines.append(".. code-block:: go")
    lines.append("")
    lines.append(f"   func {func['name']}({func['params']}) {return_type}")
    lines.append("")

    # Description
    if func["description"]:
        lines.append(func["description"])
        lines.append("")

    # Input parameters
    if func["input_params"]:
        for input_line in func["input_params"].split("\n"):
            lines.append(input_line)
        lines.append("")

    # Output parameters
    if func["output_params"]:
        lines.append(func["output_params"])
        lines.append("")

    # Example
    if func["example"]:
        # Process the example to properly format code blocks
        example_lines = func["example"].split("\n")
        in_code_block = False

        for i, line in enumerate(example_lines):
            stripped_line = line.strip()

            # Check if this is the Example: line
            if stripped_line == "Example:":
                lines.append("Example:")
                continue

            # Check if we're entering a code block
            if (
                not in_code_block
                and i > 0
                and (
                    stripped_line.startswith("import")
                    or stripped_line.startswith("if")
                    or stripped_line.startswith("for")
                )
            ):
                in_code_block = True
                lines.append("")
                lines.append(".. code-block:: go")
                lines.append("")

            # Add the line to the formatted example
            if in_code_block:
                # For code blocks, add indentation
                lines.append(f"   {line}")
            elif stripped_line:  # Only add non-empty lines outside code blocks
                lines.append(line)

        lines.append("")


def setup(app):
    """
    Setup function for Sphinx extension.
    This will be called by Sphinx when the extension is loaded.
    """
    # Register the directive
    app.add_directive("go-api-ref", GoApiRefDirective)

    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
