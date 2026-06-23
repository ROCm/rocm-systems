#!/usr/bin/env python3
"""Build script for generating test bundled binaries.

Generates various permutations of bundled binaries for testing unbundling
functionality across different platforms and compiler configurations.

Usage:
    python build_test_bundles.py [--rocm-path PATH]
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


class BundleBuilder:
    """Builder for HIP bundled binary test assets."""

    def __init__(self, rocm_path: Path | None = None):
        """Initialize the builder.

        Args:
            rocm_path: Path to ROCm installation (auto-detected if None)
        """
        self.script_dir = Path(__file__).parent
        self.kernel_src = self.script_dir / "simple_kernel.hip"
        self.host_only_src = self.script_dir / "host_only.cpp"
        self.multi_wrapper_src1 = self.script_dir / "multi_wrapper_kernel1.hip"
        self.multi_wrapper_src2 = self.script_dir / "multi_wrapper_kernel2.hip"

        # Detect platform first (needed by other methods)
        system = platform.system().lower()
        self.platform = "windows" if system == "windows" else "linux"

        # Find ROCm and tools
        self.rocm_path = self._find_rocm(rocm_path)
        self.hipcc = self._find_hipcc()
        self.clang = self._find_clang()

        # Detect code object version
        self.code_object_version = self._detect_code_object_version()

        # Setup output directory
        self.output_dir = (
            self.script_dir.parent
            / "test_assets"
            / "bundled_binaries"
            / self.platform
            / f"cov{self.code_object_version}"
        )

    def _find_rocm(self, explicit_path: Path | None) -> Path:
        """Find ROCm installation path.

        Args:
            explicit_path: Explicit ROCm path if provided

        Returns:
            Path to ROCm installation

        Raises:
            RuntimeError: If ROCm cannot be found
        """
        if explicit_path:
            if explicit_path.exists():
                return explicit_path
            raise RuntimeError(f"Specified ROCm path does not exist: {explicit_path}")

        # Check environment variables
        for env_var in ["ROCM_PATH", "HIP_PATH"]:
            env_path = os.environ.get(env_var)
            if env_path:
                path = Path(env_path)
                if path.exists():
                    return path

        raise RuntimeError(
            "Could not find ROCm installation. "
            "Please specify with --rocm-path or set ROCM_PATH/HIP_PATH environment variable."
        )

    def _find_hipcc(self) -> Path:
        """Find hipcc compiler.

        Returns:
            Path to hipcc executable

        Raises:
            RuntimeError: If hipcc cannot be found
        """
        # Try in ROCm bin directory
        hipcc_name = "hipcc.bat" if self.platform == "windows" else "hipcc"
        hipcc = self.rocm_path / "bin" / hipcc_name

        if hipcc.exists():
            return hipcc

        raise RuntimeError(f"Could not find hipcc at {hipcc}")

    def _find_clang(self) -> Path:
        """Find clang++ compiler for host-only builds.

        Returns:
            Path to clang++ executable

        Raises:
            RuntimeError: If clang++ cannot be found
        """
        # Try in ROCm LLVM bin directory
        clang_name = "clang++.exe" if self.platform == "windows" else "clang++"
        clang = self.rocm_path / "lib" / "llvm" / "bin" / clang_name

        if clang.exists():
            return clang

        # Try in ROCm llvm/bin directory (alternative layout)
        clang = self.rocm_path / "llvm" / "bin" / clang_name
        if clang.exists():
            return clang

        raise RuntimeError(
            f"Could not find clang++ in ROCm installation at {self.rocm_path}"
        )

    def _detect_code_object_version(self) -> str:
        """Detect code object version from compiler.

        Returns:
            Code object version string (e.g., "5", "6")
        """
        try:
            result = subprocess.run(
                [str(self.hipcc), "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            # Try to extract code object version from output
            for line in result.stdout.splitlines():
                if "code object version" in line.lower():
                    # Extract version number
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            return part
            # Default to 5 if not found
            return "5"
        except subprocess.CalledProcessError:
            return "5"

    def _run_hipcc(self, args: list[str], output_file: Path) -> bool:
        """Run hipcc with given arguments.

        Args:
            args: Arguments to pass to hipcc
            output_file: Output file path

        Returns:
            True if successful, False otherwise
        """
        cmd = [str(self.hipcc)] + args
        print(f"  Running: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if output_file.exists():
                size_kb = output_file.stat().st_size / 1024
                print(f"  ✓ Created: {output_file.name} ({size_kb:.1f} KB)")
                return True
            else:
                print(f"  ✗ Failed: Output file not created")
                return False
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            return False

    def _run_clang(self, args: list[str], output_file: Path) -> bool:
        """Run clang++ with given arguments.

        Args:
            args: Arguments to pass to clang++
            output_file: Output file path

        Returns:
            True if successful, False otherwise
        """
        cmd = [str(self.clang)] + args
        print(f"  Running: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if output_file.exists():
                size_kb = output_file.stat().st_size / 1024
                print(f"  ✓ Created: {output_file.name} ({size_kb:.1f} KB)")
                return True
            else:
                print(f"  ✗ Failed: Output file not created")
                return False
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            return False

    def build_exe_compressed(self):
        """Build executable with compressed bundles (if supported)."""
        exe_name = "test_kernel_compressed.exe"
        print(f"\nBuilding: {exe_name} (gfx1100,gfx1101 with compression)")
        output = self.output_dir / exe_name

        # Try with compression flag
        success = self._run_hipcc(
            [
                str(self.kernel_src),
                "--offload-arch=gfx1100",
                "--offload-arch=gfx1101",
                "-mllvm",
                "--offload-compress",
                "-o",
                str(output),
                "-fno-gpu-rdc",
                "-DBUILD_EXECUTABLE",
            ],
            output,
        )

        if not success:
            print("  ⚠ Compression may not be supported, trying without flag")
            success = self._run_hipcc(
                [
                    str(self.kernel_src),
                    "--offload-arch=gfx1100",
                    "--offload-arch=gfx1101",
                    "-o",
                    str(output),
                    "-fno-gpu-rdc",
                    "-DBUILD_EXECUTABLE",
                ],
                output,
            )

        return success

    def build_exe_wide_arch(self):
        """Build executable with wide architecture coverage."""
        exe_name = "test_kernel_wide.exe"
        print(f"\nBuilding: {exe_name} (gfx900,gfx906,gfx908,gfx90a,gfx1100)")
        output = self.output_dir / exe_name

        return self._run_hipcc(
            [
                str(self.kernel_src),
                "--offload-arch=gfx900",
                "--offload-arch=gfx906",
                "--offload-arch=gfx908",
                "--offload-arch=gfx90a",
                "--offload-arch=gfx1100",
                "-o",
                str(output),
                "-fno-gpu-rdc",
                "-DBUILD_EXECUTABLE",
            ],
            output,
        )

    def build_executable_single_arch(self):
        """Build executable with single architecture."""
        exe_name = "test_kernel_single.exe"
        print(f"\nBuilding: {exe_name} (gfx1100 executable)")
        output = self.output_dir / exe_name

        return self._run_hipcc(
            [
                str(self.kernel_src),
                "--offload-arch=gfx1100",
                "-o",
                str(output),
                "-fno-gpu-rdc",
                "-DBUILD_EXECUTABLE",
            ],
            output,
        )

    def build_executable_multi_arch(self):
        """Build executable with multiple architectures."""
        exe_name = "test_kernel_multi.exe"
        print(f"\nBuilding: {exe_name} (gfx1100,gfx1101 executable)")
        output = self.output_dir / exe_name

        return self._run_hipcc(
            [
                str(self.kernel_src),
                "--offload-arch=gfx1100",
                "--offload-arch=gfx1101",
                "-o",
                str(output),
                "-fno-gpu-rdc",
                "-DBUILD_EXECUTABLE",
            ],
            output,
        )

    def build_shared_lib_single_arch(self):
        """Build shared library with single architecture."""
        if self.platform == "windows":
            lib_name = "test_kernel_single.dll"
        else:
            lib_name = "libtest_kernel_single.so"

        print(f"\nBuilding: {lib_name} (gfx1100 shared library)")
        output = self.output_dir / lib_name

        args = [
            str(self.kernel_src),
            "--offload-arch=gfx1100",
            "-o",
            str(output),
            "-fno-gpu-rdc",
            "-shared",
        ]
        # -fPIC is required on Linux but unsupported on Windows
        if self.platform != "windows":
            args.append("-fPIC")

        return self._run_hipcc(args, output)

    def build_shared_lib_multi_arch(self):
        """Build shared library with multiple architectures."""
        if self.platform == "windows":
            lib_name = "test_kernel_multi.dll"
        else:
            lib_name = "libtest_kernel_multi.so"

        print(f"\nBuilding: {lib_name} (gfx1100,gfx1101 shared library)")
        output = self.output_dir / lib_name

        args = [
            str(self.kernel_src),
            "--offload-arch=gfx1100",
            "--offload-arch=gfx1101",
            "-o",
            str(output),
            "-fno-gpu-rdc",
            "-shared",
        ]
        # -fPIC is required on Linux but unsupported on Windows
        if self.platform != "windows":
            args.append("-fPIC")

        return self._run_hipcc(args, output)

    def build_shared_lib_multi_wrapper(self):
        """Build shared library with multiple fat binary wrappers.

        This builds two separate translation units with -fgpu-rdc (relocatable
        device code) and links them together. Each TU generates its own
        __CudaFatBinaryWrapper in .hipFatBinSegment, resulting in multiple
        wrappers in the final binary.

        This is common in large projects like RCCL that use RDC builds, and
        tests the kpack splitter's ability to transform ALL wrappers.
        """
        if self.platform == "windows":
            lib_name = "test_multi_wrapper.dll"
        else:
            lib_name = "libtest_multi_wrapper.so"

        print(f"\nBuilding: {lib_name} (RDC build with 2 wrappers)")

        # Create temp directory for intermediate object files
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            obj1 = tmpdir / "kernel1.o"
            obj2 = tmpdir / "kernel2.o"
            output = self.output_dir / lib_name

            # Compile first TU with -fgpu-rdc
            print(f"  Compiling kernel1 with -fgpu-rdc...")
            cmd1 = [
                str(self.hipcc),
                "-fgpu-rdc",
                "-c",
                str(self.multi_wrapper_src1),
                "--offload-arch=gfx1100",
                "-o",
                str(obj1),
            ]
            # -fPIC is required on Linux but unsupported on Windows
            if self.platform != "windows":
                cmd1.insert(-2, "-fPIC")
            try:
                subprocess.run(cmd1, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to compile kernel1: {e.stderr}")
                return False

            # Compile second TU with -fgpu-rdc
            print(f"  Compiling kernel2 with -fgpu-rdc...")
            cmd2 = [
                str(self.hipcc),
                "-fgpu-rdc",
                "-c",
                str(self.multi_wrapper_src2),
                "--offload-arch=gfx1100",
                "-o",
                str(obj2),
            ]
            # -fPIC is required on Linux but unsupported on Windows
            if self.platform != "windows":
                cmd2.insert(-2, "-fPIC")
            try:
                subprocess.run(cmd2, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to compile kernel2: {e.stderr}")
                return False

            # Link with -fgpu-rdc
            print(f"  Linking with -fgpu-rdc...")
            return self._run_hipcc(
                [
                    "-fgpu-rdc",
                    str(obj1),
                    str(obj2),
                    "-shared",
                    "-o",
                    str(output),
                ],
                output,
            )

    def build_host_only_executable(self):
        """Build host-only executable (no GPU device code)."""
        exe_name = "host_only.exe"
        print(f"\nBuilding: {exe_name} (host-only executable, no GPU code)")
        output = self.output_dir / exe_name

        return self._run_clang(
            [
                str(self.host_only_src),
                "-o",
                str(output),
                "-DBUILD_EXECUTABLE",
            ],
            output,
        )

    def build_host_only_shared_lib(self):
        """Build host-only shared library (no GPU device code)."""
        if self.platform == "windows":
            lib_name = "host_only.dll"
        else:
            lib_name = "libhost_only.so"

        print(f"\nBuilding: {lib_name} (host-only shared library, no GPU code)")
        output = self.output_dir / lib_name

        args = [
            str(self.host_only_src),
            "-o",
            str(output),
            "-shared",
        ]
        # -fPIC is required on Linux but unsupported on Windows
        if self.platform != "windows":
            args.append("-fPIC")

        return self._run_clang(args, output)

    def generate_manifest(self):
        """Generate manifest file documenting the test assets."""
        import datetime

        manifest_path = self.output_dir / "MANIFEST.txt"

        # Get compiler version
        try:
            result = subprocess.run(
                [str(self.hipcc), "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            compiler_version = result.stdout.splitlines()[0]
        except (subprocess.CalledProcessError, OSError):
            compiler_version = "Unknown"

        lib_prefix = "" if self.platform == "windows" else "lib"
        lib_ext = ".dll" if self.platform == "windows" else ".so"

        content = f"""Test Bundled Binaries Generated: {datetime.datetime.now()}
ROCm Path: {self.rocm_path}
Code Object Version: {self.code_object_version}
Platform: {self.platform}
Compiler: {compiler_version}

Bundled Executables (with GPU device code):
- test_kernel_single.exe: Single architecture (gfx1100)
- test_kernel_multi.exe: Multiple architectures (gfx1100, gfx1101)
- test_kernel_compressed.exe: Multiple architectures with compression (if supported)
- test_kernel_wide.exe: Wide architecture coverage (gfx900,906,908,90a,1100)

Bundled Shared Libraries (with GPU device code):
- {lib_prefix}test_kernel_single{lib_ext}: Single architecture (gfx1100)
- {lib_prefix}test_kernel_multi{lib_ext}: Multiple architectures (gfx1100, gfx1101)
- {lib_prefix}test_multi_wrapper{lib_ext}: RDC build with 2 __CudaFatBinaryWrapper structs (gfx1100)

Host-Only Binaries (NO GPU device code, for negative testing):
- host_only.exe: Host-only executable
- {lib_prefix}host_only{lib_ext}: Host-only shared library

Bundled binaries contain HIP kernels (vectorAdd, scalarMultiply) with .hip_fatbin sections.
Host-only binaries contain only CPU code with no .hip_fatbin sections.

Sources:
- test_generation/simple_kernel.hip (GPU kernels)
- test_generation/multi_wrapper_kernel1.hip (RDC kernel 1)
- test_generation/multi_wrapper_kernel2.hip (RDC kernel 2)
- test_generation/host_only.cpp (host-only code)
Build Script: test_generation/build_test_bundles.py

These assets are used to test unbundling functionality across:
- Executables and shared libraries
- Uncompressed vs compressed code objects
- Single vs multiple architectures
- Different code object versions
- ELF (Linux) vs PE/COFF (Windows) formats
- Positive tests (bundled binaries) vs negative tests (host-only binaries)
"""

        manifest_path.write_text(content)
        print(f"\n✓ Manifest created: {manifest_path}")

    def build_all(self):
        """Build all test bundles."""
        print("=" * 70)
        print("Building Test Bundled Binaries")
        print("=" * 70)
        print(f"ROCm Path: {self.rocm_path}")
        print(f"Compiler: {self.hipcc}")
        print(f"Code Object Version: {self.code_object_version}")
        print(f"Platform: {self.platform}")
        print(f"Output Directory: {self.output_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build all variants
        results = {
            # Bundled executables
            "test_kernel_single (exe)": self.build_executable_single_arch(),
            "test_kernel_multi (exe)": self.build_executable_multi_arch(),
            "test_kernel_compressed (exe)": self.build_exe_compressed(),
            "test_kernel_wide (exe)": self.build_exe_wide_arch(),
            # Bundled shared libraries
            "test_kernel_single (so/dll)": self.build_shared_lib_single_arch(),
            "test_kernel_multi (so/dll)": self.build_shared_lib_multi_arch(),
            # Multi-wrapper shared library (RDC build)
            "test_multi_wrapper (so/dll)": self.build_shared_lib_multi_wrapper(),
            # Host-only binaries (for negative testing)
            "host_only (exe)": self.build_host_only_executable(),
            "host_only (so/dll)": self.build_host_only_shared_lib(),
        }

        # Generate manifest
        self.generate_manifest()

        # Summary
        print("\n" + "=" * 70)
        print("Build Summary")
        print("=" * 70)
        for name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{status}: {name}")

        # List all generated files
        print(f"\nGenerated files in {self.output_dir}:")
        # List all binary files (executables, libraries)
        patterns = ["*.exe", "*.so", "*.dll", "test_kernel_*", "libtest_kernel_*"]
        seen_files = set()
        for pattern in patterns:
            for file in sorted(self.output_dir.glob(pattern)):
                if file.name != "MANIFEST.txt" and file not in seen_files:
                    size_kb = file.stat().st_size / 1024
                    print(f"  {file.name:40} {size_kb:8.1f} KB")
                    seen_files.add(file)

        total_success = sum(results.values())
        total_builds = len(results)
        print(f"\nTotal: {total_success}/{total_builds} successful")

        return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Build test bundled binaries for rocm-kpack testing"
    )
    parser.add_argument(
        "--rocm-path",
        type=Path,
        help="Path to ROCm installation (auto-detected if not specified)",
    )

    args = parser.parse_args()

    try:
        builder = BundleBuilder(rocm_path=args.rocm_path)
        success = builder.build_all()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Configure Windows console for UTF-8 before any output
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    from rocm_kpack.platform_utils import configure_windows_console

    configure_windows_console()

    main()
