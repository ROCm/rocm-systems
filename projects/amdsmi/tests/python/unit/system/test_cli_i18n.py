#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the CLI localization layer (``amdsmi_i18n``).

Pinned behaviour:

* with no catalog (or an English locale / ``AMDSMI_LANG=en``) every helper is an
  identity, so CLI output stays byte-identical to the untranslated tool;
* with a catalog, human-readable labels, table headers and CLI exception text
  are localized while JSON/CSV payloads (keys *and* values) stay English.

The catalog is generated in-process (no msgfmt needed) and the modules are
loaded from the CLI directory resolved by ``common.find_cli_dir``.
"""

import importlib.util
import json
import os
import re
import struct
import sys
import tempfile
import unittest

from common.common import (
    add_class_cleanup,
    cli_search_order,
    fake_module,
    find_cli_dir,
    stub_modules,
)

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
_I18N_PATH = os.path.join(_CLI_DIR, "amdsmi_i18n.py") if _CLI_DIR else None
_CYRILLIC = re.compile("[\u0400-\u04ff]")

_I18N_ENV = ("AMDSMI_LANG", "AMDSMI_LOCALEDIR", "LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG")

CATALOG = {
    "": "Content-Type: text/plain; charset=UTF-8\n",
    "label\x04POWER_USAGE": "ПОТРЕБЛЕНИЕ",
    "label\x04SOCKET_POWER": "МОЩНОСТЬ_СОКЕТА",
    "label\x04POWER": "МОЩН",
    "label\x04ACCESS TABLE": "ТАБЛИЦА ДОСТУПА",
    "Command '{command}' is invalid. Run 'amd-smi -h' for more info.": "Команда '{command}' недопустима. Справка: 'amd-smi -h'.",
    "{common_message} Error code: {value} [{error_type_name}]": "{common_message} Код ошибки: {value} [{error_type_name}]",
    "Displays output in JSON format": "Вывод в формате JSON",
}


def _write_mo(path, catalog):
    """Write a GNU .mo file (little endian, no hash table)."""
    keys = sorted(catalog)
    ids = b""
    strs = b""
    offsets = []
    for k in keys:
        kb, vb = k.encode("utf-8"), catalog[k].encode("utf-8")
        offsets.append((len(ids), len(kb), len(strs), len(vb)))
        ids += kb + b"\0"
        strs += vb + b"\0"
    n = len(keys)
    keystart = 7 * 4 + 16 * n
    valuestart = keystart + len(ids)
    koffsets, voffsets = [], []
    for o1, l1, o2, l2 in offsets:
        koffsets += [l1, o1 + keystart]
        voffsets += [l2, o2 + valuestart]
    data = struct.pack("Iiiiiii", 0x950412DE, 0, n, 7 * 4, 7 * 4 + n * 8, 0, 0)
    data += struct.pack("%di" % len(koffsets), *koffsets)
    data += struct.pack("%di" % len(voffsets), *voffsets)
    data += ids + strs
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_CLI_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _I18nCase(unittest.TestCase):
    """Reloads amdsmi_i18n (+ dependants) under a controlled environment."""

    env = {}

    @classmethod
    def setUpClass(cls):
        if not _I18N_PATH or not os.path.isfile(_I18N_PATH):
            raise unittest.SkipTest(f"amd-smi CLI amdsmi_i18n.py not found (looked in {_CLI_DIR})")
        cls._saved_env = {k: os.environ.get(k) for k in _I18N_ENV}
        add_class_cleanup(cls, cls._restore_env)
        # Fresh copies of the modules under test; the originals come back at class teardown.
        stub_modules(
            cls,
            {
                "amdsmi_i18n": None,
                "amdsmi_cli_exceptions": None,
                "amdsmi_logger_i18n_under_test": None,
                "amdsmi_helpers": fake_module(
                    "amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {})
                ),
            },
        )
        cls.tmp = tempfile.TemporaryDirectory()
        add_class_cleanup(cls, cls.tmp.cleanup)
        _write_mo(os.path.join(cls.tmp.name, "ru", "LC_MESSAGES", "amdsmi_cli.mo"), CATALOG)
        for k in _I18N_ENV:
            os.environ.pop(k, None)
        os.environ["AMDSMI_LOCALEDIR"] = cls.tmp.name
        os.environ.update(cls.env)
        cls.i18n = _load("amdsmi_i18n", "amdsmi_i18n.py")
        cls.exc = _load("amdsmi_cli_exceptions", "amdsmi_cli_exceptions.py")
        logger_mod = _load("amdsmi_logger_i18n_under_test", "amdsmi_logger.py")
        cls.logger = logger_mod.AMDSMILogger.__new__(logger_mod.AMDSMILogger)

    @classmethod
    def _restore_env(cls):
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    PAYLOAD = {"gpu": 0, "power": {"socket_power": "120 W", "power_usage": "N/A"}}
    ENGLISH = "GPU: 0\n    POWER:\n        SOCKET_POWER: 120 W\n        POWER_USAGE: N/A\n"


class TestI18nEnglish(_I18nCase):
    """English locale even though a Russian catalog exists: pure identity."""

    env = {"LANG": "en_US.UTF-8"}

    def test_inactive(self):
        self.assertFalse(self.i18n.ACTIVE)

    def test_human_readable_unchanged(self):
        self.assertEqual(self.logger._convert_json_to_human_readable(self.PAYLOAD), self.ENGLISH)

    def test_helpers_are_identity(self):
        i = self.i18n
        self.assertEqual(
            i.gettext("Displays output in JSON format"), "Displays output in JSON format"
        )
        self.assertEqual(i.header("   GPU  POWER"), "   GPU  POWER")
        s = i.lazy("Command '{command}' is invalid. Run 'amd-smi -h' for more info.", command="x")
        self.assertEqual(s, "Command 'x' is invalid. Run 'amd-smi -h' for more info.")
        self.assertEqual(i.tr(s), s)

    def test_exception_text_unchanged(self):
        e = self.exc.AmdSmiInvalidCommandException("bogus", "human_readable")
        self.assertEqual(
            str(e),
            "Command 'bogus' is invalid. Run 'amd-smi -h' for more info. "
            f"Error code: {e.value} [{e.json_message['error_type']}]",
        )


class TestI18nForcedEnglish(TestI18nEnglish):
    """AMDSMI_LANG=en overrides a Russian system locale."""

    env = {"LANG": "ru_RU.UTF-8", "AMDSMI_LANG": "en"}


class TestI18nRussian(_I18nCase):
    env = {"LANG": "ru_RU.UTF-8"}

    def setUp(self):
        if not self.i18n.ACTIVE:
            self.skipTest("stdout cannot encode Cyrillic in this environment")

    def test_labels_translated_in_human_readable(self):
        out = self.logger._convert_json_to_human_readable(self.PAYLOAD)
        self.assertEqual(
            out, "GPU: 0\n    МОЩН:\n        МОЩНОСТЬ_СОКЕТА: 120 W\n        ПОТРЕБЛЕНИЕ: N/A\n"
        )

    def test_header_keeps_column_end(self):
        self.assertEqual(self.i18n.header("GPU    POWER"), "GPU     МОЩН")
        self.assertEqual(self.i18n.header("\nACCESS TABLE"), "\nТАБЛИЦА ДОСТУПА")

    def test_exception_localized_for_humans_english_for_machines(self):
        english = "Command 'bogus' is invalid. Run 'amd-smi -h' for more info."
        human = self.exc.AmdSmiInvalidCommandException("bogus", "human_readable")
        self.assertEqual(
            str(human),
            "Команда 'bogus' недопустима. Справка: 'amd-smi -h'. "
            f"Код ошибки: {human.value} [{human.json_message['error_type']}]",
        )
        js = self.exc.AmdSmiInvalidCommandException("bogus", "json")
        self.assertEqual(json.loads(str(js))["error"], english)
        self.assertIsNone(_CYRILLIC.search(str(js)))
        csv = self.exc.AmdSmiInvalidCommandException("bogus", "csv")
        self.assertIn(english, str(csv))
        self.assertIsNone(_CYRILLIC.search(str(csv)))

    def test_lazy_is_english_str(self):
        s = self.i18n.lazy(
            "Command '{command}' is invalid. Run 'amd-smi -h' for more info.", command="x"
        )
        self.assertEqual(str(s), "Command 'x' is invalid. Run 'amd-smi -h' for more info.")
        self.assertEqual(json.dumps({"e": s}), json.dumps({"e": str(s)}))
        self.assertEqual(self.i18n.tr(s), "Команда 'x' недопустима. Справка: 'amd-smi -h'.")

    def test_argparse_errors_not_translated(self):
        # AMDSMIParser.error() parses these English messages; only help layout may change.
        self.assertEqual(
            self.i18n._argparse_gettext("unrecognized arguments: %s"), "unrecognized arguments: %s"
        )
