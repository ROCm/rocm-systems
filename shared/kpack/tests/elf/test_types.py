"""Tests for ELF type definitions."""

import pytest

from rocm_kpack.elf.types import RelaEntry, R_X86_64_RELATIVE, R_X86_64_64


class TestRelaEntryTargetAddress:
    """Tests for RelaEntry.get_target_address() and targets_range()."""

    def test_relative_relocation_target(self):
        """R_X86_64_RELATIVE: target is r_addend."""
        rela = RelaEntry(
            r_offset=0x1000,
            r_info=RelaEntry.make_info(sym=0, type_=R_X86_64_RELATIVE),
            r_addend=0x5000,
        )
        assert rela.get_target_address() == 0x5000

    def test_unknown_relocation_target(self):
        """Unknown relocation types return None."""
        rela = RelaEntry(
            r_offset=0x1000,
            r_info=RelaEntry.make_info(sym=1, type_=R_X86_64_64),
            r_addend=0x5000,
        )
        assert rela.get_target_address() is None

    def test_targets_range_inside(self):
        """Target inside range returns True."""
        rela = RelaEntry(
            r_offset=0x1000,
            r_info=RelaEntry.make_info(sym=0, type_=R_X86_64_RELATIVE),
            r_addend=0x5500,  # Inside [0x5000, 0x6000)
        )
        assert rela.targets_range(0x5000, 0x1000) is True

    def test_targets_range_at_start(self):
        """Target at range start returns True."""
        rela = RelaEntry(
            r_offset=0x1000,
            r_info=RelaEntry.make_info(sym=0, type_=R_X86_64_RELATIVE),
            r_addend=0x5000,  # At start of [0x5000, 0x6000)
        )
        assert rela.targets_range(0x5000, 0x1000) is True

    def test_targets_range_at_end(self):
        """Target at range end (exclusive) returns False."""
        rela = RelaEntry(
            r_offset=0x1000,
            r_info=RelaEntry.make_info(sym=0, type_=R_X86_64_RELATIVE),
            r_addend=0x6000,  # At end of [0x5000, 0x6000) - exclusive
        )
        assert rela.targets_range(0x5000, 0x1000) is False

    def test_targets_range_before(self):
        """Target before range returns False."""
        rela = RelaEntry(
            r_offset=0x1000,
            r_info=RelaEntry.make_info(sym=0, type_=R_X86_64_RELATIVE),
            r_addend=0x4000,  # Before [0x5000, 0x6000)
        )
        assert rela.targets_range(0x5000, 0x1000) is False

    def test_targets_range_after(self):
        """Target after range returns False."""
        rela = RelaEntry(
            r_offset=0x1000,
            r_info=RelaEntry.make_info(sym=0, type_=R_X86_64_RELATIVE),
            r_addend=0x7000,  # After [0x5000, 0x6000)
        )
        assert rela.targets_range(0x5000, 0x1000) is False

    def test_targets_range_unknown_type(self):
        """Unknown relocation type returns None."""
        rela = RelaEntry(
            r_offset=0x1000,
            r_info=RelaEntry.make_info(sym=1, type_=R_X86_64_64),
            r_addend=0x5500,  # Would be inside range if we understood it
        )
        assert rela.targets_range(0x5000, 0x1000) is None
