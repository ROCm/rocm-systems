# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Profile contract for CMPX writes to EXEC and VCC."""

import pytest

from amdisa.isa_profile import (
    Cdna1Profile,
    Cdna2Profile,
    Cdna4Profile,
    CdnaProfile,
    Rdna1Profile,
    Rdna2Profile,
    Rdna3Profile,
    Rdna3_5Profile,
    Rdna4Profile,
)

_CDNA_PROFILES = [CdnaProfile, Cdna1Profile, Cdna2Profile, Cdna4Profile]
_RDNA_PROFILES = [
    Rdna1Profile,
    Rdna2Profile,
    Rdna3Profile,
    Rdna3_5Profile,
    Rdna4Profile,
]


class TestCmpxWritesVccProfileContract:
    """V_CMPX writes EXEC + VCC on CDNA (GFX9) and only EXEC on RDNA."""

    @pytest.mark.parametrize('profile_cls', _CDNA_PROFILES)
    def test_cdna_profiles_write_vcc(self, profile_cls):
        assert profile_cls().cmpx_writes_vcc is True, (
            f'{profile_cls.__name__}.cmpx_writes_vcc must be True: CDNA '
            f"V_CMPX writes both EXEC and VCC (or SDST under VOP3). If you "
            f"are intentionally redefining this property, also update "
            f"the compare binding."
        )

    @pytest.mark.parametrize('profile_cls', _RDNA_PROFILES)
    def test_rdna_profiles_do_not_write_vcc(self, profile_cls):
        assert profile_cls().cmpx_writes_vcc is False, (
            f'{profile_cls.__name__}.cmpx_writes_vcc must be False: RDNA '
            f'V_CMPX writes only EXEC.'
        )
