#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Minimal gettext-based localization for the amd-smi CLI.

Language selection (first match wins):
  1. AMDSMI_LANG - optional override, e.g. AMDSMI_LANG=ru or AMDSMI_LANG=en.
     "en*", "C" and "POSIX" force the original English text.
  2. Standard gettext variables: LANGUAGE, LC_ALL, LC_MESSAGES, LANG.

Catalogs live in <cli dir>/locale/<lang>/LC_MESSAGES/amdsmi_cli.mo
(override the directory with AMDSMI_LOCALEDIR).  When no catalog is found the
module is a pure identity layer, so output is byte-identical to the English CLI.

Only human-facing text is translated.  Machine-readable output (JSON/CSV keys and
values) must never pass through this module; for strings that may end up in both
(e.g. CLI exception messages) use lazy() which yields an English ``str`` and
carries the localized text in ``.tr`` for human-readable display only.
"""

import gettext as _gettext
import os
import re
import sys

DOMAIN = "amdsmi_cli"
LOCALEDIR = os.environ.get("AMDSMI_LOCALEDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "locale"
)


def _select_languages():
    """Return the language list for gettext, or [] to force English."""
    override = os.environ.get("AMDSMI_LANG", "").strip()
    if override:
        lang = override.split(".")[0].split("@")[0]
        if lang in ("C", "POSIX") or lang.lower().startswith("en"):
            return []
        return [override, lang]
    return None  # let gettext consult LANGUAGE, LC_ALL, LC_MESSAGES, LANG


def _stdout_can_encode(sample="\u0439"):
    """The CLI strips non-ASCII when stdout cannot encode it; avoid empty output."""
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None) or "ascii"
        try:
            sample.encode(enc)
        except (UnicodeEncodeError, LookupError):
            return False
    return True


def _load():
    languages = _select_languages()
    if languages == []:
        return _gettext.NullTranslations()
    try:
        trans = _gettext.translation(DOMAIN, LOCALEDIR, languages=languages, fallback=True)
    except Exception:  # never let localization break the tool
        return _gettext.NullTranslations()
    if type(trans) is not _gettext.NullTranslations and not _stdout_can_encode():
        return _gettext.NullTranslations()
    return trans


_translations = _load()
ACTIVE = type(_translations) is not _gettext.NullTranslations


def gettext(message):
    if not ACTIVE:
        return message
    return _translations.gettext(message)


def ngettext(singular, plural, n):
    if not ACTIVE:
        return singular if n == 1 else plural
    return _translations.ngettext(singular, plural, n)


def pgettext(context, message):
    if not ACTIVE:
        return message
    return _translations.pgettext(context, message)


_ = gettext


def N_(message):
    """Mark a string for extraction without translating it (stays English)."""
    return message


class LazyStr(str):
    """An English ``str`` that also carries its localized rendering in ``.tr``.

    Behaves exactly like the English string everywhere (JSON, CSV, comparisons,
    f-strings), so it is safe for machine-readable output.  Human-readable paths
    call tr(value) to display the localized form.
    """

    __slots__ = ("tr",)

    def __new__(cls, english, localized=None):
        obj = super().__new__(cls, english)
        obj.tr = english if localized is None else localized
        return obj

    def __add__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return LazyStr(str(self) + str(other), self.tr + tr(other))

    def __radd__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return LazyStr(str(other) + str(self), tr(other) + self.tr)


def lazy(template, **kwargs):
    """English str with localized companion.

    ``lazy("Value '{arg}' is invalid", arg=x)`` == ``f"Value '{x}' is invalid"``.
    With no kwargs the template is used verbatim (no str.format call).
    """
    if kwargs:
        english = template.format(**kwargs)
    else:
        english = template
    if not ACTIVE:
        return LazyStr(english)
    localized_template = _translations.gettext(template)
    if kwargs:
        try:
            localized = localized_template.format(**{k: tr(v) for k, v in kwargs.items()})
        except (KeyError, IndexError, ValueError):  # broken translation: fall back
            localized = english
    else:
        localized = localized_template
    return LazyStr(english, localized)


def tr(value):
    """Localized form of a LazyStr (or the value unchanged)."""
    if ACTIVE and isinstance(value, LazyStr):
        return value.tr
    return value


# ---------------------------------------------------------------------------
# Display labels (human-readable / table output only)
# ---------------------------------------------------------------------------
LABEL_CONTEXT = "label"
_KEY_RE = re.compile(r"^(\s*)(\S.*?)(\s*)$")


def label(text):
    """Translate an uppercase display label (e.g. "MAX_PCIE_SPEED")."""
    if not ACTIVE or not text:
        return text
    return _translations.pgettext(LABEL_CONTEXT, text)


def label_key_part(key_part):
    """Translate the key column of a human-readable "KEY: value" line,
    preserving its indentation."""
    if not ACTIVE:
        return key_part
    m = _KEY_RE.match(key_part)
    if not m:
        return key_part
    lead, key, trail = m.groups()
    translated = _translations.pgettext(LABEL_CONTEXT, key)
    if translated == key:
        return key_part
    return lead + translated + trail


_TOKEN_RE = re.compile(r"\S+")


def header(text):
    """Translate a fixed-width table header/title token by token.

    Each translated token is right-aligned inside the column span of the
    original token when it fits; longer translations widen the line.
    """
    if not ACTIVE or not text:
        return text
    m = _KEY_RE.match(text)
    if m and m.group(2):
        whole = _translations.pgettext(LABEL_CONTEXT, m.group(2))
        if whole != m.group(2):
            return m.group(1) + whole + m.group(3)
    out = []
    pos = 0
    for m in _TOKEN_RE.finditer(text):
        start, end = m.span()
        token = m.group(0)
        translated = _translations.pgettext(LABEL_CONTEXT, token)
        out.append(text[pos:start])
        if translated == token:
            out.append(token)
        elif len(translated) <= len(token):
            out.append(translated.rjust(len(token)))
        else:
            out.append(translated)
        pos = end
    out.append(text[pos:])
    return "".join(out)


# argparse strings that are safe to localize.  argparse *error* messages are
# deliberately excluded: AMDSMIParser.error() parses them ("invalid choice",
# "unrecognized arguments", "expected ...") to pick exception types/exit codes,
# and they end up in JSON/CSV error output.
ARGPARSE_HELP_MSGIDS = frozenset(
    (
        "usage: ",
        "options",
        "optional arguments",
        "positional arguments",
        "show this help message and exit",
        "show program's version number and exit",
    )
)


def _argparse_gettext(message):
    if message in ARGPARSE_HELP_MSGIDS:
        return gettext(message)
    return message


def install_argparse():
    """Route argparse's help-layout strings through this catalog.
    No-op unless a translation is active."""
    if not ACTIVE:
        return
    import argparse

    argparse._ = _argparse_gettext


install_argparse()
