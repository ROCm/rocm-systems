# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""S_SLEEP derives as `true_nop` and gets a real delay from the generator.

The MR ISA gives S_SLEEP no pseudocode, and the delay *is* the instruction --
there is no result register -- so a body that only yields leaves it with no
architectural effect and lets a sleep loop spin at the speed of its own scalar
code. The delay lived in a hand-edit of the checked-in generated header once,
which meant the next regeneration reverted it. Nothing in the C++ ISA harness
catches that (it lists s_sleep in SKIP_PREFIXES); the only guard is
gdb.rocm/lane-info.exp, hours away in the expect suite. These tests are the
near guard.
"""

from types import SimpleNamespace

from amdisa.codegen import CodeGenerator


def _body(name: str) -> str:
    """Emitted execute() body for a sleep op, straight from the generator."""
    generator = CodeGenerator.__new__(CodeGenerator)
    sem = SimpleNamespace(
        name=name,
        semantic_class='true_nop',
        operation=None,
        data_type=None,
        sets_scc=None,
        branch_condition=None,
    )
    return CodeGenerator._sleep_body(generator, sem)


class TestSleepBody:
    def test_sleep_charges_the_wave_for_the_delay(self):
        body = _body('S_SLEEP')
        assert 'wf.set_sleep_cycles(' in body
        assert 'kSleepClocksPerUnit = 64' in body

    def test_sleep_takes_its_count_from_the_literal(self):
        body = _body('S_SLEEP')
        assert 'simm16.encoding_value_' in body
        assert '0x7Fu' in body

    def test_sleep_var_takes_its_count_from_a_scalar(self):
        """S_SLEEP_VAR is the same delay with the 7-bit count in an operand."""
        body = _body('S_SLEEP_VAR')
        assert 'read_scalar(ssrc0)' in body
        assert '0x7Fu' in body
        assert 'wf.set_sleep_cycles(' in body

    def test_both_still_yield_to_the_event_loop(self):
        """FUNCTIONAL execution has to return so peer CUs can make progress."""
        for name in ('S_SLEEP', 'S_SLEEP_VAR'):
            assert 'wf.cu().request_functional_yield();' in _body(name), name
