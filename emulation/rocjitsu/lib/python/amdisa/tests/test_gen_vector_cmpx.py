# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests guarding the V_CMPX write-back contract.

Two concerns are covered here, in two layers:

1. **Profile contract** (:class:`TestCmpxWritesVccProfileContract`):
   asserts the ground-truth value of ``cmpx_writes_vcc`` on every real
   ``IsaProfile`` subclass. This catches override drift — e.g., a refactor
   that flips the base default and forgets to re-pin a CDNA override.

2. **Codegen wiring** (:class:`TestGenVectorCmpxWriteBacks`): asserts
   that ``CodeGenerator._gen_vector_cmpx`` honours the profile's
   ``cmpx_writes_vcc`` for both the VCC fallback and the VOP3 SDST
   write paths. Parametrized as a truth table over
   (profile, is_vop3, has_dst) so the structural invariants are visible.

Layer 1 catches "the property no longer reports the right value"; layer 2
catches "the codegen no longer consults the property correctly". Both
must hold for V_CMPX SGPR writes to actually reach the emitted C++.
"""

import re

import pytest

from amdisa.codegen import CodeGenerator
from amdisa.isa_profile import (
    Cdna1Profile,
    Cdna2Profile,
    CdnaProfile,
    Rdna1Profile,
    Rdna2Profile,
    Rdna3Profile,
    Rdna3_5Profile,
    Rdna4Profile,
)


# ---------------------------------------------------------------------------
# Layer 1: profile contract
# ---------------------------------------------------------------------------

_CDNA_PROFILES = [CdnaProfile, Cdna1Profile, Cdna2Profile]
_RDNA_PROFILES = [
    Rdna1Profile, Rdna2Profile, Rdna3Profile, Rdna3_5Profile, Rdna4Profile,
]


class TestCmpxWritesVccProfileContract:
    """V_CMPX writes EXEC + VCC on CDNA (GFX9) and only EXEC on RDNA."""

    @pytest.mark.parametrize('profile_cls', _CDNA_PROFILES)
    def test_cdna_profiles_write_vcc(self, profile_cls):
        assert profile_cls().cmpx_writes_vcc is True, (
            f'{profile_cls.__name__}.cmpx_writes_vcc must be True: CDNA '
            f"V_CMPX writes both EXEC and VCC (or SDST under VOP3). If you "
            f"are intentionally redefining this property, also update "
            f"CodeGenerator._gen_vector_cmpx."
        )

    @pytest.mark.parametrize('profile_cls', _RDNA_PROFILES)
    def test_rdna_profiles_do_not_write_vcc(self, profile_cls):
        assert profile_cls().cmpx_writes_vcc is False, (
            f'{profile_cls.__name__}.cmpx_writes_vcc must be False: RDNA '
            f'V_CMPX writes only EXEC.'
        )


# ---------------------------------------------------------------------------
# Layer 2: codegen wiring
# ---------------------------------------------------------------------------

class _StubSpec:
    """Holds a real IsaProfile under ``.profile`` for codegen consumption."""

    def __init__(self, profile) -> None:
        self.profile = profile


def _make_codegen(profile) -> CodeGenerator:
    """Build a CodeGenerator without running its real __init__.

    ``_gen_vector_cmpx`` only reads ``self.isa_spec.profile.cmpx_writes_vcc``
    on the codegen instance when ``op == 't'`` (which short-circuits before
    ``_cmp_condition`` runs). A hollow object suffices.

    NOTE: any attribute access added to that path will surface here as an
    ``AttributeError`` — that's a feature, not a bug; failures should
    prompt re-evaluation of the test fixture.
    """
    cg = object.__new__(CodeGenerator)
    cg.isa_spec = _StubSpec(profile)
    return cg


# Regex tolerant to whitespace and identifier choice. We anchor on the
# stable parts of the contract — the API call (``write_scalar64``,
# ``set_vcc``, ``set_exec``) and the dst identifier — not on the exact
# variable name of the result-mask local.
def _re_write_scalar64(dst_ident: str) -> re.Pattern[str]:
    return re.compile(rf'\b{re.escape(dst_ident)}\s*\.\s*write_scalar64\s*\(\s*wf\s*,\s*\w+\s*\)\s*;')


_RE_SET_VCC = re.compile(r'\bwf\s*\.\s*set_vcc\s*\(\s*\w+\s*\)\s*;')
_RE_SET_EXEC = re.compile(r'\bwf\s*\.\s*set_exec\s*\(\s*\w+\s*\)\s*;')
_RE_ANY_WRITE_SCALAR64 = re.compile(r'\.\s*write_scalar64\s*\(')


# Truth table driving the codegen tests. Each row pins one cell of the
# (profile, is_vop3, has_dst) invariant: EXEC is always written; the
# scalar-write target depends on the profile property and the encoding.
_DST_IDENT = 's_sdst_pair'

_CASES = [
    # (id, profile_factory, is_vop3, dst, expect_sdst_write, expect_vcc_write)
    pytest.param(CdnaProfile,  True,  [_DST_IDENT], True,  False, id='cdna-vop3-with-dst'),
    pytest.param(CdnaProfile,  False, None,         False, True,  id='cdna-vopc-no-dst'),
    pytest.param(CdnaProfile,  True,  None,         False, True,  id='cdna-vop3-no-dst'),
    pytest.param(CdnaProfile,  False, [_DST_IDENT], False, True,  id='cdna-vopc-with-dst'),
    pytest.param(Rdna4Profile, True,  [_DST_IDENT], False, False, id='rdna-vop3-with-dst'),
    pytest.param(Rdna4Profile, False, None,         False, False, id='rdna-vopc-no-dst'),
    pytest.param(Rdna4Profile, True,  None,         False, False, id='rdna-vop3-no-dst'),
    pytest.param(Rdna4Profile, False, [_DST_IDENT], False, False, id='rdna-vopc-with-dst'),
]


class TestGenVectorCmpxWriteBacks:
    """Truth-table coverage of the ``_gen_vector_cmpx`` write-back paths.

    Asserts the structural invariant rather than exact string output:

      * ``set_exec`` is emitted unconditionally.
      * ``write_scalar64`` on the SDST is emitted iff
        ``profile.cmpx_writes_vcc and is_vop3 and dst``.
      * ``set_vcc`` is emitted iff
        ``profile.cmpx_writes_vcc and not (is_vop3 and dst)``.

    The ``op='t'`` shortcut is intentional: it bypasses ``_cmp_condition``
    so the test stays focused on the write-back logic the user is worried
    about, and avoids depending on ``self._has_abs``.
    """

    @pytest.mark.parametrize(
        'profile_cls, is_vop3, dst, expect_sdst_write, expect_vcc_write', _CASES
    )
    def test_writeback_invariants(
        self, profile_cls, is_vop3, dst, expect_sdst_write, expect_vcc_write
    ):
        cg = _make_codegen(profile_cls())
        body = cg._gen_vector_cmpx(
            src=['s0', 's1'],
            op='t',  # bypasses _cmp_condition; see class docstring
            dtype='u32',
            is_vop3=is_vop3,
            dst=dst,
        )

        # EXEC is always updated.
        assert _RE_SET_EXEC.search(body), (
            f'V_CMPX must always update EXEC; not found in:\n{body}'
        )

        # SDST write: must appear iff expected, and target the right ident.
        sdst_match = _re_write_scalar64(_DST_IDENT).search(body) if dst else None
        sdst_count = len(_RE_ANY_WRITE_SCALAR64.findall(body))
        if expect_sdst_write:
            assert sdst_match, (
                f'Expected write_scalar64 on {_DST_IDENT!r}; body was:\n{body}'
            )
            # Guard against a stray second write_scalar64 to a different
            # target slipping through alongside the expected one.
            assert sdst_count == 1, (
                f'Expected exactly one write_scalar64 call, got '
                f'{sdst_count}; body was:\n{body}'
            )
        else:
            assert sdst_count == 0, (
                f'Did not expect any write_scalar64 call, got '
                f'{sdst_count}; body was:\n{body}'
            )

        # VCC write: must appear iff expected.
        vcc_match = _RE_SET_VCC.search(body)
        if expect_vcc_write:
            assert vcc_match, (
                f'Expected wf.set_vcc(...) call; body was:\n{body}'
            )
        else:
            assert not vcc_match, (
                f'Did not expect wf.set_vcc(...) call; body was:\n{body}'
            )

        # Mutual exclusion: a single V_CMPX never writes both VCC and SDST.
        assert not (
            (sdst_match is not None) and (vcc_match is not None)
        ), f'V_CMPX wrote both SDST and VCC; body was:\n{body}'
