# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The trap-handler control ops derive as `true_nop` and get real bodies.

The MR ISA carries no pseudocode for S_RFE / S_SENDMSG / S_SENDMSGHALT, so
they derive as `true_nop`. They are not nops: the configured GPU trap handler
returns through S_RFE and reports through S_SENDMSG, and leaving these empty
disables ROCgdb's whole stop/resume path. The bodies therefore live in the
generator, and these tests pin the parts of them that a regeneration has
previously been able to silently drop.
"""

from types import SimpleNamespace

from amdisa.codegen import CodeGenerator


def _body(name: str) -> str:
    """Emitted execute() body for a trap-control op, straight from the generator."""
    generator = CodeGenerator.__new__(CodeGenerator)
    sem = SimpleNamespace(
        name=name,
        semantic_class='true_nop',
        operation=None,
        data_type=None,
        sets_scc=None,
        branch_condition=None,
    )
    return CodeGenerator._trap_control_body(generator, sem)


class TestSRfeBody:
    def test_restores_the_saved_pc(self):
        body = _body('S_RFE_B64')
        assert 'read_scalar64(ssrc0)' in body
        assert 'kPcAddressMask' in body

    def test_restores_the_interrupted_exec(self):
        """Regression: a handler that returns without stopping the wave used to
        leave its own EXEC mask installed, silently un-diverging a branch for
        the rest of the kernel. gdb.rocm/lane-info.exp catches it, as does
        WaveDebugTest.TrapHandlerRestoresInterruptedExecWhenReturningWithoutStopping.
        This block once existed only in the checked-in generated header, so the
        next regeneration would have removed it."""
        body = _body('S_RFE_B64')
        assert 'wf.set_exec(wf.trap_saved_exec());' in body
        assert 'if (wf.in_trap_handler())' in body
        # The restore has to happen while the flag still says we are in the
        # handler, so it must precede the clear.
        assert body.index('wf.set_exec(wf.trap_saved_exec());') < body.index(
            'wf.set_in_trap_handler(false);'
        )

    def test_honours_status_halt_on_the_way_out(self):
        body = _body('S_RFE_B64')
        assert 'kStatusHalt' in body
        assert 'wf.set_debug_halted(true);' in body

    def test_every_spelling_of_the_return_gets_a_body(self):
        """GFX1250 spells it S_RFE_I64 (SOP1 opcode 74). Omitting a spelling
        leaves that ISA's generated execute_impl() an empty no-op, so its trap
        handlers can never return."""
        for name in ('S_RFE', 'S_RFE_B64', 'S_RFE_I64'):
            body = _body(name)
            assert 'kPcAddressMask' in body, f'{name} has no trap-return body'
            assert 'wf.set_exec(wf.trap_saved_exec());' in body, name

    def test_gfx12_restores_application_status_and_gfx1250_restores_the_full_pc(self):
        body = _body('S_RFE_I64')
        assert '0x01FFFFFFFFFFFFFFULL' in body
        assert 'wf.trap_saved_status()' in body
        assert 'restored_status | kStatusHalt' in body

        older_body = _body('S_RFE_B64')
        assert '0x0000FFFFFFFFFFFFULL' in older_body
        assert 'wf.trap_saved_status()' in older_body
        assert 'wf.uses_separate_trap_ctrl()' in older_body


class TestSendmsgBody:
    def test_reports_msg_interrupt_from_inside_the_handler(self):
        body = _body('S_SENDMSG')
        assert 'wf.set_trap_interrupt_sent(true);' in body
        assert 'wf.cu().handle_sendmsg(wf, message);' in body

    def test_sendmsghalt_halts_through_architectural_status(self):
        """S_SENDMSGHALT halts the wave -- that is the instruction, not a
        debugger artefact. Setting only the debugger's private flag left the
        two halves disagreeing: the s_rfe body reads STATUS.HALT to decide
        whether the wave stays stopped, and saw 0 for a wave this instruction
        had already halted."""
        body = _body('S_SENDMSGHALT')
        assert 'kStatusHalt' in body
        assert 'wf.set_status_raw(' in body
        assert 'wf.set_debug_halted(true);' in body

    def test_sendmsghalt_records_that_it_raised_the_halt(self):
        """The trap handler raises STATUS.HALT too, with s_setreg just before it
        returns, and there the bit means "keep the wave stopped" -- the opposite
        of what it means here. The CWSR record cannot tell the two apart, so the
        resume path in SimulatedKfd::apply_cwsr_to_wave reads this marker to
        decide whether clearing HALT loses a breakpoint or unsticks a wave."""
        body = _body('S_SENDMSGHALT')
        assert 'wf.set_self_halted(true);' in body

    def test_plain_sendmsg_does_not_halt(self):
        body = _body('S_SENDMSG')
        assert 'set_debug_halted' not in body
        assert 'set_self_halted' not in body


class TestSelfHaltedProvenance:
    def test_rfe_clears_a_stale_marker_when_nothing_is_halting_the_wave(self):
        """Leaving the marker set past a resume would make the *next* stop clear
        a STATUS.HALT the trap handler raised, silently losing that breakpoint."""
        body = _body('S_RFE_B64')
        assert 'wf.set_self_halted(false);' in body
        # Only on the not-halted path: a wave staying stopped keeps its
        # provenance, which is what the resume needs to read.
        assert body.index('wf.set_debug_halted(true);') < body.index(
            'wf.set_self_halted(false);'
        )
