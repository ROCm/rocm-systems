"""Inline grouped memberdefs into file XMLs after Doxygen runs.

When a header uses @defgroup / @addtogroup, Doxygen hoists the full
<memberdef> into group__NAME.xml and leaves the file XML with only
<member refid="..."> pointers. Breathe's doxygenfunction lookup needs
the full memberdef in a file or namespace compound, so it silently fails
for every grouped member ("Cannot find function ...").

This extension copies each grouped memberdef into the file XML matching
its <location file=...>, restoring what the XML would look like without
@defgroup. Idempotent if rerun.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

from sphinx.util import logging

logger = logging.getLogger(__name__)


def _inline(xml_dir: Path) -> int:
    if not xml_dir.exists():
        return 0

    file_xmls = {}
    for fx in xml_dir.glob("*_8h.xml"):
        try:
            tree = ET.parse(fx)
        except ET.ParseError:
            continue
        cd = tree.getroot().find("compounddef")
        if cd is None:
            continue
        loc = cd.find("location")
        if loc is None or not loc.get("file"):
            continue
        file_xmls[Path(loc.get("file")).name] = (fx, tree, cd)

    by_file = {}
    for gx in sorted(xml_dir.glob("group__*.xml")):
        try:
            gtree = ET.parse(gx)
        except ET.ParseError:
            continue
        for md in gtree.getroot().iter("memberdef"):
            loc = md.find("location")
            if loc is None or not loc.get("file"):
                continue
            header = Path(loc.get("file")).name
            if header in file_xmls:
                by_file.setdefault(header, []).append(md)

    inlined = 0
    for header, members in by_file.items():
        fx_path, tree, cd = file_xmls[header]
        already = any(
            sd.get("kind") == "user-defined"
            and (sd.findtext("header") or "").startswith("Grouped members")
            for sd in cd.findall("sectiondef")
        )
        if already:
            continue
        section = ET.SubElement(cd, "sectiondef")
        section.set("kind", "user-defined")
        hdr = ET.SubElement(section, "header")
        hdr.text = "Grouped members"
        for m in members:
            section.append(m)
        tree.write(fx_path, xml_declaration=True, encoding="UTF-8")
        inlined += len(members)

    return inlined


def _on_builder_inited(app):
    xml_dir = Path(app.confdir) / "doxygen" / "xml"
    n = _inline(xml_dir)
    if n:
        logger.info(
            "inline_doxygen_groups: inlined %d memberdefs from group XMLs", n
        )


def setup(app):
    # Priority 600 ensures this runs after rocm-docs-core's Doxygen
    # invocation (default priority 500).
    app.connect("builder-inited", _on_builder_inited, priority=600)
    return {"version": "1.0", "parallel_read_safe": True}
