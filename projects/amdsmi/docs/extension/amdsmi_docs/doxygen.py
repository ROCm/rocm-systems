"""
Register custom Sphinx directives for rendering AMD SMI API documentation.

This extension reads a Doxygen tag file, collects API symbols and groups, and
emits Breathe directives to generate API reference content in Sphinx. It
supports Myst-flavoured Markdown sources via `myst-parser`.
"""

import xml.etree.ElementTree as XMLTree
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.environment import BuildEnvironment
from sphinx.errors import ExtensionError
from sphinx.util.docutils import SphinxDirective


@dataclass(slots=True)
class DoxygenTagData:
    # groups are represented by `@defgroup` and their members by `@ingroup`
    # in Doxygen comments
    groups: list[tuple[str, str]] = field(default_factory=list)
    enums: list[str] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)
    structs: list[str] = field(default_factory=list)
    unions: list[str] = field(default_factory=list)
    typedefs: list[str] = field(default_factory=list)


def _render_myst_options(options: Sequence[str]) -> str:
    return "".join(
        f"{option if option.startswith(':') else f':{option}:'}\n"
        for option in options
    )


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def render_groups_myst_markdown(
    groups: Sequence[tuple[str, str]], heading_level: int
) -> str:
    hashes = "#" * heading_level
    return "\n\n".join(
        f"({tag})=\n\n"
        f"{hashes} {title}\n\n"
        f"```{{doxygengroup}} {tag}\n"
        f":content-only:\n"
        f"```"
        for tag, title in groups
    )


def render_members_myst_markdown(
    directive_name: str,
    names: Sequence[str],
    options: Sequence[str] = (),
) -> str:
    breathe_options = _render_myst_options(options)

    return "\n\n".join(
        f"({name})=\n\n"
        f"```{{{directive_name}}} {name}\n"
        f"{breathe_options}"
        f"```"
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

        ```{amdsmi-doxygen}
        :kind: enums
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
        kind = self.options["kind"].strip().lower()

        if kind != "groups" and "heading-level" in self.options:
            raise self.error(":heading-level: is only valid when :kind: is 'groups'.")

        heading_level = self.options.get("heading-level", 2)
        if kind == "groups" and not 1 <= heading_level <= 6:
            raise self.error(":heading-level: must be between 1 and 6.")

        data = self._get_doxygen_tag_data()

        match kind:
            case "groups":
                return self._render_groups(data.groups, heading_level)
            case "enums":
                return self._render_members("doxygenenum", data.enums)
            case "defines":
                return self._render_members("doxygendefine", data.defines)
            case "structs":
                return self._render_members(
                    "doxygenstruct",
                    data.structs,
                    (":members:",),
                )
            case "unions":
                return self._render_members(
                    "doxygenunion",
                    data.unions,
                    (":members:",),
                )
            case "typedefs":
                return self._render_members("doxygentypedef", data.typedefs)
            case _:
                raise self.error(
                    f"Invalid :kind: {kind!r}. Expected one of: "
                    "groups, enums, defines, structs, unions, typedefs"
                )

    def _render_members(
        self,
        directive_name: str,
        names: Sequence[str],
        options: Sequence[str] = (),
    ) -> list[nodes.Node]:
        if not names:
            return []

        text = render_members_myst_markdown(directive_name, names, options)
        return self.parse_text_to_nodes(text)

    def _render_groups(
        self,
        groups: Sequence[tuple[str, str]],
        heading_level: int,
    ) -> list[nodes.Node]:
        if not groups:
            return []

        text = render_groups_myst_markdown(groups, heading_level)
        return self.parse_text_to_nodes(text, allow_section_headings=True)

    def _get_doxygen_tag_data(self) -> DoxygenTagData:
        tagfile = Path(self.config.amdsmi_doxygen_tagfile).resolve()
        self.env.note_dependency(str(tagfile))

        cache = getattr(self.env, "_amdsmi_doxygen_tag_cache", None)
        if cache is None:
            cache = {}
            setattr(self.env, "_amdsmi_doxygen_tag_cache", cache)

        key = str(tagfile)
        data = cache.get(key)
        if data is not None:
            return data

        if not tagfile.exists():
            raise self.error(
                f"Doxygen tagfile not found: {tagfile}. "
                "Check the GENERATE_TAGFILE option in Doxyfile and "
                "'amdsmi_doxygen_tagfile' in conf.py."
            )

        try:
            root = XMLTree.parse(tagfile).getroot()
        except XMLTree.ParseError as e:
            raise self.error(f"Failed to parse Doxygen tagfile: {tagfile}") from e

        data = DoxygenTagData()

        for compound in root.findall("compound"):
            match compound.get("kind"):
                case "group":
                    group_tag = (compound.findtext("name") or "").strip()
                    group_title = (compound.findtext("title") or "").strip()
                    if group_tag and group_title:
                        data.groups.append((group_tag, group_title))

                case "file":
                    for member in compound.findall("member"):
                        member_kind = member.get("kind")
                        member_name = (member.findtext("name") or "").strip()
                        if not member_name:
                            continue

                        if member_kind == "enumeration":
                            data.enums.append(member_name)
                        elif member_kind == "define":
                            data.defines.append(member_name)
                        elif member_kind == "typedef":
                            data.typedefs.append(member_name)

                case "struct":
                    struct_name = (compound.findtext("name") or "").strip()
                    if struct_name:
                        data.structs.append(struct_name)

                case "union":
                    union_name = (compound.findtext("name") or "").strip()
                    if union_name:
                        data.unions.append(union_name)

                case _:
                    continue

        data.groups = list(dict.fromkeys(data.groups))
        data.enums = _dedupe_preserve_order(data.enums)
        data.defines = _dedupe_preserve_order(data.defines)
        data.structs = _dedupe_preserve_order(data.structs)
        data.unions = _dedupe_preserve_order(data.unions)
        data.typedefs = _dedupe_preserve_order(data.typedefs)

        cache[key] = data
        return data


def _validate_breathe_loaded(app: Sphinx, _config: Config) -> None:
    if "breathe" not in app.extensions:
        raise ExtensionError(
            "The 'amdsmi-doxygen' extension requires the 'breathe' Sphinx "
            "extension to be enabled in conf.py."
        )


def _clear_tagfile_cache(
    _app: Sphinx,
    env: BuildEnvironment,
    _docnames: list[str],
) -> None:
    if hasattr(env, "_amdsmi_doxygen_tag_cache"):
        delattr(env, "_amdsmi_doxygen_tag_cache")


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_config_value(
        "amdsmi_doxygen_tagfile",
        Path(app.confdir) / "doxygen" / "_out" / "tagfile.xml",
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
