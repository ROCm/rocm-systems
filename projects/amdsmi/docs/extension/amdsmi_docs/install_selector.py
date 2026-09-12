"""Selector for the install page.

Content is emitted visible and is hidden by the browser script, so the page
still shows every method when JavaScript is unavailable.

Sections are tagged with a marker element rather than wrapped in a container,
because docutils does not allow a section heading inside a directive body.
"""

from __future__ import annotations

import json
from typing import Any

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

# axis id -> (legend, [(value, label), ...])
AXES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "method": (
        "Installation method",
        [
            ("rocm", "ROCm Core SDK"),
            ("standalone", "Standalone package"),
            ("nightly", "Nightly build"),
            ("tarball", "Tarball"),
            ("pypi", "PyPI wheel"),
        ],
    ),
    "distro": (
        "Linux distribution",
        [
            ("debian", "Debian / Ubuntu"),
            ("rhel", "RHEL / Fedora"),
            ("sles", "SLES / openSUSE"),
        ],
    ),
}

# An axis only worth showing for some methods; the script hides it otherwise.
AXIS_APPLIES_TO: dict[str, list[str]] = {"distro": ["standalone"]}

_OPTION_SPEC = {axis: directives.unchanged for axis in AXES}


def _selection(options: dict[str, str]) -> dict[str, list[str]]:
    """Parse directive options into axis -> accepted values."""
    return {axis: value.split() for axis, value in options.items() if value.split()}


def _attr(selection: dict[str, list[str]]) -> str:
    return json.dumps(selection, sort_keys=True).replace('"', "&quot;")


class InstallWhen(SphinxDirective):
    """Wrap body content that applies only to certain selections.

    Emits a plain container and encodes the selection in class names, so every
    writer -- including the llms.txt Markdown one -- keeps the content.
    """

    has_content = True
    option_spec = _OPTION_SPEC

    def run(self) -> list[nodes.Node]:
        node = nodes.container()
        node["classes"] = ["amdsmi-when"] + [
            f"amdsmi-when--{axis}-{value}"
            for axis, values in _selection(self.options).items()
            for value in values
        ]
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class InstallSection(SphinxDirective):
    """Tag the enclosing section so the script can show or hide it."""

    has_content = False
    option_spec = _OPTION_SPEC

    def run(self) -> list[nodes.Node]:
        marker = (
            '<div class="amdsmi-section-marker" hidden '
            f'data-selector="{_attr(_selection(self.options))}"></div>'
        )
        return [nodes.raw("", marker, format="html")]


class InstallSelector(SphinxDirective):
    """Render the selector control panel."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        config = json.dumps(
            {
                "axes": {axis: [v for v, _ in opts] for axis, (_, opts) in AXES.items()},
                "appliesTo": AXIS_APPLIES_TO,
            },
            sort_keys=True,
        ).replace('"', "&quot;")

        parts = [f'<div class="amdsmi-selector" data-amdsmi-selector="{config}">']
        for axis, (legend, options) in AXES.items():
            parts.append(f'<div class="amdsmi-selector__axis" data-axis="{axis}">')
            parts.append(f'<span class="amdsmi-selector__legend">{legend}</span>')
            parts.append(f'<div role="group" aria-label="{legend}" class="amdsmi-selector__options">')
            for value, label in options:
                parts.append(
                    '<button type="button" class="amdsmi-selector__option" '
                    f'data-value="{value}" aria-pressed="false">{label}</button>'
                )
            parts.append("</div></div>")
        parts.append(
            '<noscript><p class="amdsmi-selector__noscript">JavaScript is disabled, '
            "so every installation method is shown below.</p></noscript>"
        )
        parts.append("</div>")
        return [nodes.raw("", "".join(parts), format="html")]


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_directive("install-selector", InstallSelector)
    app.add_directive("install-section", InstallSection)
    app.add_directive("install-when", InstallWhen)
    app.add_css_file("install-selector.css")
    app.add_js_file("install-selector.js")
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
