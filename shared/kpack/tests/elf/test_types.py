"""Tests for ELF type definitions."""

import pytest

from rocm_kpack.elf.types import (
    RelaEntry,
    ArchConfig,
    get_arch_config,
    R_X86_64_RELATIVE,
    R_X86_64_64,
    EM_X86_64,
)


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

    def test_get_target_address_with_explicit_r_relative(self):
        """get_target_address honors an explicitly provided r_relative value."""
        FAKE_R_RELATIVE = 0xDEAD
        rela = RelaEntry(
            r_offset=0x1000,
            r_info=RelaEntry.make_info(sym=0, type_=FAKE_R_RELATIVE),
            r_addend=0x5000,
        )
        assert rela.get_target_address(r_relative=FAKE_R_RELATIVE) == 0x5000
        # Default r_relative (R_X86_64_RELATIVE) should not match
        assert rela.get_target_address() is None

    def test_targets_range_with_explicit_r_relative(self):
        """targets_range honors an explicitly provided r_relative value."""
        FAKE_R_RELATIVE = 0xDEAD
        rela = RelaEntry(
            r_offset=0x1000,
            r_info=RelaEntry.make_info(sym=0, type_=FAKE_R_RELATIVE),
            r_addend=0x5500,
        )
        assert rela.targets_range(0x5000, 0x1000, r_relative=FAKE_R_RELATIVE) is True
        # Default r_relative should treat this as an unknown type
        assert rela.targets_range(0x5000, 0x1000) is None


class TestArchConfig:
    """Tests for ArchConfig and get_arch_config()."""

    def test_x86_64_config(self):
        """get_arch_config returns correct values for x86_64."""
        config = get_arch_config(EM_X86_64)
        assert isinstance(config, ArchConfig)
        assert config.page_size == 0x1000
        assert config.r_relative == R_X86_64_RELATIVE

    def test_unknown_machine_raises(self):
        """get_arch_config raises ValueError for unsupported machine types."""
        UNKNOWN_MACHINE = 0xFFFF
        with pytest.raises(ValueError, match="Unsupported ELF machine type"):
            get_arch_config(UNKNOWN_MACHINE)

    def test_unknown_machine_error_includes_machine_value(self):
        """ValueError message includes the unknown machine type value."""
        UNKNOWN_MACHINE = 9999
        with pytest.raises(ValueError, match=str(UNKNOWN_MACHINE)):
            get_arch_config(UNKNOWN_MACHINE)
