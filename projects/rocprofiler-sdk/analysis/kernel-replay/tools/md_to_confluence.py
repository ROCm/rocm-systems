#!/usr/bin/env python3
"""Convert a study markdown file to Confluence storage-format XHTML.

Usage:
    python3 md_to_confluence.py INPUT.md [INPUT2.md ...] > out.html

Handles the constructs used by the kernel-replay study: ATX headings, tables,
fenced code blocks, inline code, emphasis, and lists. Fenced code becomes a
Confluence ``code`` macro; LaTeX-ish math in the source is passed through as
literal text, since the design-doc pages do not render math macros.
"""

import html
import re
import sys

import markdown


def fence_to_macro(match: "re.Match[str]") -> str:
    language = (match.group("lang") or "text").strip() or "text"
    body = match.group("body")
    return (
        '<ac:structured-macro ac:name="code">'
        f'<ac:parameter ac:name="language">{html.escape(language)}</ac:parameter>'
        f"<ac:plain-text-body><![CDATA[{body}]]></ac:plain-text-body>"
        "</ac:structured-macro>\n"
    )


FENCE = re.compile(
    r"^```(?P<lang>[^\n`]*)\n(?P<body>.*?)^```\s*$",
    re.DOTALL | re.MULTILINE,
)


def convert(text: str) -> str:
    placeholders: list[str] = []

    def stash(match: "re.Match[str]") -> str:
        placeholders.append(fence_to_macro(match))
        return f"\n\nKRSTUDYFENCE{len(placeholders) - 1}\n\n"

    text = FENCE.sub(stash, text)

    # Confluence pages here do not render math; strip the delimiters so the
    # formulas remain readable as plain text rather than showing as markup.
    text = text.replace("$$", "").replace("\\boxed{", "").replace("$", "`")

    body = markdown.markdown(text, extensions=["tables", "sane_lists", "attr_list"])

    for index, macro in enumerate(placeholders):
        body = body.replace(f"<p>KRSTUDYFENCE{index}</p>", macro)
    return body


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    parts = []
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as handle:
            parts.append(convert(handle.read()))
    print("\n<hr/>\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
