"""Unit tests for database handlers."""

from pathlib import Path

import pytest

from rocm_kpack.database_handlers import (
    RocBLASHandler,
    HipBLASLtHandler,
    HipSparseLtHandler,
    AotritonHandler,
    MIOpenHandler,
    WHEEL_TYPE_PRESETS,
    get_database_handlers,
    list_available_handlers,
)


class TestRocBLASHandler:
    """Tests for RocBLASHandler detection logic."""

    @pytest.fixture
    def handler(self):
        return RocBLASHandler()

    @pytest.fixture
    def prefix_root(self, tmp_path):
        """Create a temporary prefix root directory."""
        root = tmp_path / "prefix"
        root.mkdir()
        return root

    def test_name(self, handler):
        """Test handler name."""
        assert handler.name() == "rocblas"

    def test_detect_co_file(self, handler, prefix_root):
        """Test detection of .co file in rocblas/library."""
        file_path = prefix_root / "lib/rocblas/library/TensileLibrary_gfx1100.co"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1100"

    def test_detect_hsaco_file(self, handler, prefix_root):
        """Test detection of .hsaco file in rocblas/library."""
        file_path = prefix_root / "lib/rocblas/library/kernel_gfx1101.hsaco"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1101"

    def test_detect_dat_file(self, handler, prefix_root):
        """Test detection of .dat file in rocblas/library."""
        file_path = prefix_root / "lib/rocblas/library/TensileLibrary_gfx1102.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1102"

    def test_detect_various_architectures(self, handler, prefix_root):
        """Test detection of various gfx architecture formats."""
        test_cases = [
            ("TensileLibrary_gfx90a.dat", "gfx90a"),
            ("TensileLibrary_gfx942.dat", "gfx942"),
            ("kernels_gfx1030.co", "gfx1030"),
            ("TensileLibrary_lazy_gfx90a-xnack+.hsaco", "gfx90a-xnack+"),
            ("Kernels.so-000-gfx942-xnack-.hsaco", "gfx942-xnack-"),
        ]

        for filename, expected_arch in test_cases:
            file_path = prefix_root / f"lib/rocblas/library/{filename}"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

            result = handler.detect(file_path, prefix_root)
            assert result == expected_arch, f"Failed for {filename}"

    def test_detect_xnack_plus(self, handler, prefix_root):
        """Test detection of xnack+ suffix in filename."""
        file_path = (
            prefix_root
            / "lib/rocblas/library/TensileLibrary_Type_HH_gfx90a-xnack+.hsaco"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx90a-xnack+"

    def test_detect_xnack_minus(self, handler, prefix_root):
        """Test detection of xnack- suffix in filename."""
        file_path = (
            prefix_root / "lib/rocblas/library/Kernels.so-000-gfx942-xnack-.hsaco"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx942-xnack-"

    def test_reject_wrong_directory(self, handler, prefix_root):
        """Test that files not in rocblas/library are rejected."""
        file_path = prefix_root / "lib/other/library/TensileLibrary_gfx1100.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_wrong_extension(self, handler, prefix_root):
        """Test that files with unsupported extensions are rejected."""
        file_path = prefix_root / "lib/rocblas/library/TensileLibrary_gfx1100.txt"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_no_architecture(self, handler, prefix_root):
        """Test that files without architecture suffix are rejected."""
        file_path = prefix_root / "lib/rocblas/library/TensileLibrary.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_file_outside_prefix(self, handler, prefix_root):
        """Files outside prefix root are a caller bug — must raise."""
        file_path = Path("/tmp/rocblas/library/TensileLibrary_gfx1100.dat")

        with pytest.raises(ValueError, match="is not under prefix_root"):
            handler.detect(file_path, prefix_root)


class TestHipBLASLtHandler:
    """Tests for HipBLASLtHandler detection logic."""

    @pytest.fixture
    def handler(self):
        return HipBLASLtHandler()

    @pytest.fixture
    def prefix_root(self, tmp_path):
        """Create a temporary prefix root directory."""
        root = tmp_path / "prefix"
        root.mkdir()
        return root

    def test_name(self, handler):
        """Test handler name."""
        assert handler.name() == "hipblaslt"

    def test_detect_co_file(self, handler, prefix_root):
        """Test detection of .co file in hipblaslt/library."""
        file_path = prefix_root / "lib/hipblaslt/library/TensileLibrary_gfx1100.co"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1100"

    def test_detect_hsaco_file(self, handler, prefix_root):
        """Test detection of .hsaco file in hipblaslt/library."""
        file_path = prefix_root / "lib/hipblaslt/library/kernel_gfx1101.hsaco"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1101"

    def test_detect_dat_file(self, handler, prefix_root):
        """Test detection of .dat file in hipblaslt/library."""
        file_path = prefix_root / "lib/hipblaslt/library/TensileLibrary_gfx1102.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx1102"

    def test_detect_various_architectures(self, handler, prefix_root):
        """Test detection of various gfx architecture formats."""
        test_cases = [
            ("TensileLibrary_gfx90a.dat", "gfx90a"),
            ("TensileLibrary_gfx942.dat", "gfx942"),
            ("kernels_gfx1030.co", "gfx1030"),
            ("TensileLibrary_lazy_gfx90a-xnack+.hsaco", "gfx90a-xnack+"),
            ("Kernels.so-000-gfx942-xnack-.hsaco", "gfx942-xnack-"),
        ]

        for filename, expected_arch in test_cases:
            file_path = prefix_root / f"lib/hipblaslt/library/{filename}"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

            result = handler.detect(file_path, prefix_root)
            assert result == expected_arch, f"Failed for {filename}"

    def test_detect_xnack_plus(self, handler, prefix_root):
        """Test detection of xnack+ suffix in filename."""
        file_path = (
            prefix_root
            / "lib/hipblaslt/library/TensileLibrary_Type_HH_gfx90a-xnack+.hsaco"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx90a-xnack+"

    def test_detect_xnack_minus(self, handler, prefix_root):
        """Test detection of xnack- suffix in filename."""
        file_path = (
            prefix_root / "lib/hipblaslt/library/Kernels.so-000-gfx942-xnack-.hsaco"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx942-xnack-"

    def test_reject_wrong_directory(self, handler, prefix_root):
        """Test that files not in hipblaslt/library are rejected."""
        file_path = prefix_root / "lib/rocblas/library/TensileLibrary_gfx1100.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_wrong_extension(self, handler, prefix_root):
        """Test that files with unsupported extensions are rejected."""
        file_path = prefix_root / "lib/hipblaslt/library/TensileLibrary_gfx1100.json"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_no_architecture(self, handler, prefix_root):
        """Test that files without architecture suffix are rejected."""
        file_path = prefix_root / "lib/hipblaslt/library/generic.dat"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_file_outside_prefix(self, handler, prefix_root):
        """Files outside prefix root are a caller bug — must raise."""
        file_path = Path("/tmp/hipblaslt/library/TensileLibrary_gfx1100.dat")

        with pytest.raises(ValueError, match="is not under prefix_root"):
            handler.detect(file_path, prefix_root)


class TestHipSparseLtHandler:
    """Tests for HipSparseLtHandler detection logic."""

    @pytest.fixture
    def handler(self):
        return HipSparseLtHandler()

    @pytest.fixture
    def prefix_root(self, tmp_path):
        root = tmp_path / "prefix"
        root.mkdir()
        return root

    def test_name(self, handler):
        assert handler.name() == "hipsparselt"

    def test_detect_co_file(self, handler, prefix_root):
        file_path = prefix_root / "lib/hipsparselt/library/extop_gfx942.co"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx942"

    def test_detect_hsaco_file(self, handler, prefix_root):
        file_path = prefix_root / "lib/hipsparselt/library/Kernels.so-000-gfx950.hsaco"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx950"

    def test_detect_dat_file(self, handler, prefix_root):
        file_path = (
            prefix_root / "lib/hipsparselt/library/TensileLibrary_BB_BB_A_gfx942.dat"
        )
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx942"

    def test_reject_wrong_directory(self, handler, prefix_root):
        file_path = prefix_root / "lib/rocblas/library/something_gfx942.co"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_wrong_extension(self, handler, prefix_root):
        file_path = prefix_root / "lib/hipsparselt/library/something_gfx942.so"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None


class TestAotritonHandler:
    """Tests for AotritonHandler detection logic."""

    @pytest.fixture
    def handler(self):
        return AotritonHandler()

    @pytest.fixture
    def prefix_root(self, tmp_path):
        """Create a temporary prefix root directory."""
        root = tmp_path / "prefix"
        root.mkdir()
        return root

    def test_name(self, handler):
        """Test handler name."""
        assert handler.name() == "aotriton"

    def test_detect_aks2_file(self, handler, prefix_root):
        """Test detection of .aks2 file in aotriton.images layout."""
        file_path = (
            prefix_root / "lib/aotriton.images/amd-gfx942/flash/attn_fwd/kernel.aks2"
        )
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx942"

    def test_detect_family_arch(self, handler, prefix_root):
        """Test that family architecture names are mapped to bundle keys."""
        file_path = (
            prefix_root / "lib/aotriton.images/amd-gfx11xx/flash/attn_fwd/kernel.aks2"
        )
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx11"

    def test_detect_various_architectures(self, handler, prefix_root):
        """Test detection of all real aotriton.images architecture directories.

        Family/sub-family patterns are mapped to bundle keys:
            gfx11xx → gfx11, gfx120x → gfx12_0.
        Target names pass through unchanged.
        """
        test_cases = [
            ("amd-gfx90a", "gfx90a"),
            ("amd-gfx942", "gfx942"),
            ("amd-gfx950", "gfx950"),
            ("amd-gfx11xx", "gfx11"),
            ("amd-gfx120x", "gfx12_0"),
        ]

        for arch_dir, expected_bundle_key in test_cases:
            file_path = (
                prefix_root
                / f"lib/aotriton.images/{arch_dir}/flash/attn_fwd/kernel.aks2"
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

            result = handler.detect(file_path, prefix_root)
            assert result == expected_bundle_key, f"Failed for {arch_dir}"

    def test_detect_signature_file(self, handler, prefix_root):
        """Test that __signature__ file under arch dir is detected."""
        file_path = prefix_root / "lib/aotriton.images/amd-gfx942/__signature__"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx942"

    def test_detect_deeply_nested_file(self, handler, prefix_root):
        """Test detection of file nested deep under architecture directory."""
        file_path = (
            prefix_root
            / "lib/aotriton.images/amd-gfx950/flash/bwd_kernel_dk_dv/kernel.aks2"
        )
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx950"

    def test_reject_no_amd_prefix(self, handler, prefix_root):
        """Test that directories without amd- prefix are rejected."""
        file_path = prefix_root / "lib/aotriton.images/gfx942/flash/kernel.aks2"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_non_gfx_directory(self, handler, prefix_root):
        """Test that non-gfx directories under aotriton.images are rejected."""
        file_path = prefix_root / "lib/aotriton.images/amd-common/flash/kernel.aks2"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_file_outside_prefix(self, handler, prefix_root):
        """Files outside prefix root are a caller bug — must raise."""
        file_path = Path("/tmp/aotriton.images/amd-gfx942/kernel.aks2")

        with pytest.raises(ValueError, match="is not under prefix_root"):
            handler.detect(file_path, prefix_root)

    def test_reject_wrong_parent_directory(self, handler, prefix_root):
        """Test that aotriton without .images suffix is rejected."""
        file_path = prefix_root / "lib/aotriton/amd-gfx942/flash/kernel.aks2"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None


class TestMIOpenHandler:
    """Tests for MIOpenHandler detection logic."""

    @pytest.fixture
    def handler(self):
        return MIOpenHandler()

    @pytest.fixture
    def prefix_root(self, tmp_path):
        """Create a temporary prefix root directory."""
        root = tmp_path / "prefix"
        root.mkdir()
        return root

    def test_name(self, handler):
        assert handler.name() == "miopen"

    def test_detect_tn_model(self, handler, prefix_root):
        file_path = prefix_root / "share/miopen/db/gfx908.tn.model"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx908"

    def test_detect_ktn_model(self, handler, prefix_root):
        file_path = (
            prefix_root
            / "share/miopen/db/gfx942_ConvHipIgemmGroupFwdXdlops_decoder.ktn.model"
        )
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx942"

    def test_detect_metadata_model(self, handler, prefix_root):
        file_path = prefix_root / "share/miopen/db/gfx90a_metadata.tn.model"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx90a"

    def test_detect_3d_model(self, handler, prefix_root):
        file_path = prefix_root / "share/miopen/db/gfx950_3d.tn.model"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx950"

    def test_detect_db_txt(self, handler, prefix_root):
        """MIOpen .db.txt files with concatenated arch+CU count."""
        file_path = prefix_root / "share/miopen/db/gfx90878.db.txt"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx908"

    def test_detect_fdb_txt(self, handler, prefix_root):
        """MIOpen .HIP.fdb.txt files."""
        file_path = prefix_root / "share/miopen/db/gfx942130.HIP.fdb.txt"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx942"

    def test_detect_opencl_fdb_txt(self, handler, prefix_root):
        file_path = prefix_root / "share/miopen/db/gfx900_56.OpenCL.fdb.txt"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result == "gfx900"

    def test_detect_various_architectures(self, handler, prefix_root):
        test_cases = [
            ("gfx908.tn.model", "gfx908"),
            ("gfx90a_ConvHipIgemmGroupXdlops_encoder.ktn.model", "gfx90a"),
            ("gfx942_3d_metadata.tn.model", "gfx942"),
            ("gfx950_ConvHipImplicitGemm3DGroupFwdXdlops_metadata.tn.model", "gfx950"),
            ("gfx90878.db.txt", "gfx908"),
            ("gfx90a68.db.txt", "gfx90a"),
            ("gfx942130.HIP.fdb.txt", "gfx942"),
            ("gfx1030_36.db.txt", "gfx1030"),
            ("gfx906_60.OpenCL.fdb.txt", "gfx906"),
            ("gfx90a6e.HIP.fdb.txt", "gfx90a"),
            ("gfx942e4.db.txt", "gfx942"),
        ]

        for filename, expected_arch in test_cases:
            file_path = prefix_root / f"share/miopen/db/{filename}"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

            result = handler.detect(file_path, prefix_root)
            assert result == expected_arch, f"Failed for {filename}"

    def test_reject_wrong_directory(self, handler, prefix_root):
        file_path = prefix_root / "share/other/db/gfx908.tn.model"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_wrong_extension(self, handler, prefix_root):
        file_path = prefix_root / "share/miopen/db/gfx908.json"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_no_architecture(self, handler, prefix_root):
        file_path = prefix_root / "share/miopen/db/generic.tn.model"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = handler.detect(file_path, prefix_root)
        assert result is None

    def test_reject_file_outside_prefix(self, handler, prefix_root):
        """Files outside prefix root are a caller bug — must raise."""
        file_path = Path("/tmp/share/miopen/db/gfx908.tn.model")

        with pytest.raises(ValueError, match="is not under prefix_root"):
            handler.detect(file_path, prefix_root)


class TestDatabaseHandlerRegistry:
    """Tests for database handler registry functions."""

    def test_list_available_handlers(self):
        """Test that list_available_handlers returns all registered handlers."""
        handlers = list_available_handlers()
        assert isinstance(handlers, list)
        assert "rocblas" in handlers
        assert "hipblaslt" in handlers
        assert "hipsparselt" in handlers
        assert "aotriton" in handlers
        assert "miopen" in handlers
        assert len(handlers) == 5

    def test_get_database_handlers_single(self):
        """Test getting a single handler by name."""
        handlers = get_database_handlers(["rocblas"])
        assert len(handlers) == 1
        assert isinstance(handlers[0], RocBLASHandler)

    def test_get_database_handlers_multiple(self):
        """Test getting multiple handlers by name."""
        handlers = get_database_handlers(["rocblas", "hipblaslt"])
        assert len(handlers) == 2
        assert isinstance(handlers[0], RocBLASHandler)
        assert isinstance(handlers[1], HipBLASLtHandler)

    def test_get_database_handlers_all(self):
        """Test getting all handlers."""
        handlers = get_database_handlers(
            ["rocblas", "hipblaslt", "hipsparselt", "aotriton", "miopen"]
        )
        assert len(handlers) == 5
        assert isinstance(handlers[0], RocBLASHandler)
        assert isinstance(handlers[1], HipBLASLtHandler)
        assert isinstance(handlers[2], HipSparseLtHandler)
        assert isinstance(handlers[3], AotritonHandler)
        assert isinstance(handlers[4], MIOpenHandler)

    def test_wheel_type_preset(self):
        """Test that wheel type presets resolve to valid handlers."""
        names = WHEEL_TYPE_PRESETS["torch-fat"]
        handlers = get_database_handlers(names)
        assert len(handlers) == 5

    def test_get_database_handlers_unknown(self):
        """Test that unknown handler name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown database handler: unknown"):
            get_database_handlers(["unknown"])

    def test_get_database_handlers_mixed_valid_invalid(self):
        """Test that partially invalid handler list raises ValueError."""
        with pytest.raises(ValueError, match="Unknown database handler: invalid"):
            get_database_handlers(["rocblas", "invalid"])
