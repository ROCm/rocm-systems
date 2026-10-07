# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DS_GWS_* derives as true_nop and retires through one shared helper.

The nop class reports the instruction unimplemented, which halts the wave
before the kernel can finish. The helper warns once: retiring the op is not a
dispatch-wide barrier.
"""

from types import SimpleNamespace

from amdisa.codegen import CodeGenerator
from amdisa.semantics import derive_semantics

_GWS_NAMES = (
    'DS_GWS_SEMA_RELEASE_ALL',
    'DS_GWS_INIT',
    'DS_GWS_SEMA_V',
    'DS_GWS_SEMA_BR',
    'DS_GWS_SEMA_P',
    'DS_GWS_BARRIER',
)


def _body(name: str) -> str:
    generator = CodeGenerator.__new__(CodeGenerator)
    sem = SimpleNamespace(
        name=name,
        semantic_class='true_nop',
        operation=None,
        data_type=None,
        sets_scc=None,
        branch_condition=None,
    )
    return CodeGenerator._gws_body(generator, sem)


class TestGwsBody:
    def test_gws_derives_as_true_nop(self):
        for name in _GWS_NAMES:
            sem = derive_semantics(name, 'ENC_DS')
            assert sem is not None, name
            assert sem.semantic_class == 'true_nop', name

    def test_gws_body_retires_through_the_shared_helper(self):
        for name in _GWS_NAMES:
            assert _body(name) == '  amdgpu::retire_global_wave_sync(wf);', name

    def test_ordered_count_stays_unimplemented(self):
        sem = derive_semantics('DS_ORDERED_COUNT', 'ENC_DS')
        assert sem is not None
        assert sem.semantic_class == 'nop'
