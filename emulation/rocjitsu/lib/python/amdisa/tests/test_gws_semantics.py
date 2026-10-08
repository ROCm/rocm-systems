# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DS_GWS_INIT retires; the GWS barrier and semaphore ops stay unimplemented.

GWS state is not modeled. Retiring INIT unblocks the runtime's initialization
dispatch. Retiring a barrier or semaphore wait would let a kernel continue
before the grid has synchronized.
"""

from amdisa.semantics import derive_semantics

_SYNCHRONIZING_GWS_NAMES = (
    'DS_GWS_SEMA_RELEASE_ALL',
    'DS_GWS_SEMA_V',
    'DS_GWS_SEMA_BR',
    'DS_GWS_SEMA_P',
    'DS_GWS_BARRIER',
)


class TestGwsSemantics:
    def test_gws_init_derives_as_true_nop(self):
        sem = derive_semantics('DS_GWS_INIT', 'ENC_DS')
        assert sem is not None
        assert sem.semantic_class == 'true_nop'

    def test_gws_synchronization_stays_unimplemented(self):
        for name in _SYNCHRONIZING_GWS_NAMES:
            sem = derive_semantics(name, 'ENC_DS')
            assert sem is not None, name
            assert sem.semantic_class == 'nop', name

    def test_ordered_count_stays_unimplemented(self):
        sem = derive_semantics('DS_ORDERED_COUNT', 'ENC_DS')
        assert sem is not None
        assert sem.semantic_class == 'nop'
