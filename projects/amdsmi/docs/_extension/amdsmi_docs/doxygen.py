"""
Register custom Sphinx directives for rendering AMD SMI API documentation.

This extension reads a Doxygen tag file, collects API symbols and groups, and
emits Breathe directives to generate API reference content in Sphinx. It
supports Myst-flavoured Markdown sources via `myst-parser` and
reStructuredText.
"""

import xml.etree.ElementTree as XMLTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.environment import BuildEnvironment
from sphinx.errors import ConfigError, ExtensionError
from sphinx.util.docutils import SphinxDirective

RST_HEADING_CHARS: Final[dict[int, str]] = {
    1: "=",
    2: "-",
    3: "~",
    4: "^",
    5: '"',
    6: "+",
}


@dataclass(slots=True)
class DoxygenTagData:
    # groups are represented by `@defgroup` and its members by `@ingroup` in Doxygen comments
    groups: list[tuple[str, str]] = field(default_factory=list)
    enums: list[str] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)
    structs: list[str] = field(default_factory=list)
    unions: list[str] = field(default_factory=list)
    typedefs: list[str] = field(default_factory=list)


def render_groups_markdown(groups: list[tuple[str, str]], heading_level: int) -> str:
    hashes = "#" * heading_level
    return "\n\n".join(
        # use raw HTML for the anchor because I don't want Myst's `()=` syntax
        # to format the anchor text
        f"<span id='{tag}'></span>\n\n"
        + f"{hashes} {title}\n\n"
        + f"```{{doxygengroup}} {tag}\n"
        + ":content-only:\n```"
        for tag, title in groups
    )


def render_groups_rst(groups: list[tuple[str, str]], heading_level: int) -> str:
    adornment = RST_HEADING_CHARS[heading_level]
    return "\n\n".join(
        ".. raw:: html\n\n"
        + f"   <span id='{tag}'></span>"
        + f"{title}\n"
        + f"{adornment * len(title)}\n\n"
        + f".. doxygengroup:: {tag}\n"
        + f"   :content-only:"
        for tag, title in groups
    )


def render_members_markdown(directive_name: str, names: list[str]) -> str:
    return "\n\n".join(
        f"<span id='{name}'></span>\n\n```{{{directive_name}}} {name}\n```"
        for name in names
    )


def render_members_rst(directive_name: str, names: list[str]) -> str:
    return "\n\n".join(
        f".. raw:: html\n\n   <span id='{name}'></span>\n\n.. {directive_name}:: {name}"
        for name in names
    )


class AmdsmiDoxygenDirective(SphinxDirective):
    """
    Render AMD SMI API content from a Doxygen tagfile via Breathe.

    Supported kinds:
    - groups
    - enums
    - defines
    - structs
    - unions
    - typedefs

    Usage in .md files:

        ```{amdsmi-doxygen}
        :kind: groups
        :heading-level: 2
        ```
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        "kind": directives.unchanged_required,
        "heading-level": directives.nonnegative_int,
    }

    def run(self) -> list[nodes.Node]:
        kind_opt = self.options.get("kind")
        if not kind_opt:
            raise self.error("The :kind: option is required.")
        kind = kind_opt.strip().lower()

        heading_level = self.options.get("heading-level", 2)
        if not 1 <= heading_level <= 6:
            raise self.error(":heading-level: must be between 1 and 6.")

        if kind != "groups" and "heading-level" in self.options:
            raise self.error(":heading-level: is only valid when :kind: is 'groups'.")

        data = self._get_doxygen_tag_data()

        match kind:
            case "groups":
                return self._render_groups(data.groups, heading_level)
            case "enums":
                return self._render_members("doxygenenum", data.enums)
            case "defines":
                return self._render_members("doxygendefine", data.defines)
            case "structs":
                return self._render_members("doxygenstruct", data.structs)
            case "unions":
                return self._render_members("doxygenunion", data.unions)
            case "typedefs":
                return self._render_members("doxygentypedef", data.typedefs)
            case _:
                raise self.error(
                    f"Invalid :kind: {kind!r}. Expected one of:\n."
                    + f"groups, enums, defines, structs, unions, typedefs"
                )

    def _is_rst_source(self) -> bool:
        source, _line = self.get_source_info()
        suffix = Path(source).suffix.lower() if source else ""
        return suffix == ".rst"

    def _render_members(
        self, directive_name: str, names: list[str]
    ) -> list[nodes.Node]:
        if not names:
            return []

        if self._is_rst_source():
            text = render_members_rst(directive_name, names)
        else:
            text = render_members_markdown(directive_name, names)

        return self.parse_text_to_nodes(text)

    def _render_groups(
        self, groups: list[tuple[str, str]], heading_level: int
    ) -> list[nodes.Node]:
        if not groups:
            return []

        if self._is_rst_source():
            text = render_groups_rst(groups, heading_level)
        else:
            text = render_groups_markdown(groups, heading_level)

        return self.parse_text_to_nodes(text, allow_section_headings=True)

    def _get_doxygen_tag_data(self) -> DoxygenTagData:
        tagfile = Path(self.config.amdsmi_doxygen_tagfile)
        self.env.note_dependency(tagfile)

        cache = getattr(self.env, "_amdsmi_doxygen_tag_cache", None)
        if cache is None:
            cache = {}
            setattr(self.env, "_amdsmi_doxygen_tag_cache", cache)

        key = str(tagfile)
        data = cache.get(key)
        if data is not None:
            return data

        try:
            root = XMLTree.parse(tagfile).getroot()
        except FileNotFoundError as e:
            raise ConfigError(
                f"Doxygen tagfile not found: {tagfile}\n"
                + f"Check the GENERATE_TAGFILE option in Doxyfile and set 'amdsmi_doxygen_tagfile' in your Sphinx configuration"
            ) from e
        except XMLTree.ParseError as e:
            raise ConfigError(f"Failed to parse Doxygen tag file: {tagfile}") from e

        data = DoxygenTagData()

        for compound in root.findall("compound"):
            match compound.get("kind"):
                case "group":
                    group_tag = compound.findtext("name")
                    group_title = compound.findtext("title")
                    if group_tag and group_title:
                        data.groups.append((group_tag, group_title))

                case "file":
                    for member in compound.findall("member"):
                        member_kind = member.get("kind")
                        member_name = member.findtext("name")
                        if not member_name:
                            continue

                        if member_kind == "enumeration":
                            data.enums.append(member_name)
                        elif member_kind == "define":
                            data.defines.append(member_name)
                        elif member_kind == "typedef":
                            data.typedefs.append(member_name)

                case "struct":
                    struct_name = compound.findtext("name")
                    if struct_name:
                        data.structs.append(struct_name)

                case "union":
                    union_name = compound.findtext("name")
                    if union_name:
                        data.unions.append(union_name)

                case _:
                    continue

        cache[key] = data
        return data


def _validate_breathe_loaded(app: Sphinx, _config: Config) -> None:
    if "breathe" not in app.extensions:
        raise ExtensionError(
            "The 'amdsmi-doxygen' extension requires the 'breathe' Sphinx "
            + "extension to be enabled in conf.py."
        )


def _clear_tagfile_cache(
    _app: Sphinx, env: BuildEnvironment, _docnames: list[str]
) -> None:
    if hasattr(env, "_amdsmi_doxygen_tag_cache"):
        delattr(env, "_amdsmi_doxygen_tag_cache")


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_config_value(
        "amdsmi_doxygen_tagfile",
        Path(app.confdir) / "doxygen" / "tagfile.xml",
        "env",
        types=(str, Path),
    )
    app.add_directive("amdsmi-doxygen", AmdsmiDoxygenDirective)
    app.connect("config-inited", _validate_breathe_loaded, priority=600)
    app.connect("env-before-read-docs", _clear_tagfile_cache)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
