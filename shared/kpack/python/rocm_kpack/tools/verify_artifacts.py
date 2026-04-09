#!/usr/bin/env python3

"""
Tool for verifying split artifacts meet expected invariants.

This tool validates that artifact splitting produced correct output:
- Fat binaries converted to host-only (PROGBITS -> NOBITS)
- Architecture separation is correct
- Kpack archives are valid
- Manifest files are present and valid
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import msgpack

from rocm_kpack.binutils import Toolchain, has_section, get_section_type


@dataclass
class VerificationResult:
    """Result of a single verification check."""

    check_name: str
    passed: bool
    message: str
    details: list[str]


class ArtifactVerifier:
    """Verifies split artifacts meet expected invariants."""

    def __init__(
        self, artifacts_dir: Path, toolchain: Toolchain, verbose: bool = False
    ):
        self.artifacts_dir = artifacts_dir
        self.toolchain = toolchain
        self.verbose = verbose
        self.results: list[VerificationResult] = []
        self.errors = 0
        self.warnings = 0

    def run_all_checks(self) -> bool:
        """Run all verification checks. Returns True if all pass."""
        print("=" * 70)
        print("ARTIFACT VERIFICATION")
        print("=" * 70)
        print(f"Artifacts directory: {self.artifacts_dir}\n")

        # Discover artifacts
        artifacts = self._discover_artifacts()
        if not artifacts:
            self._fail(
                "No artifacts found",
                f"No artifact directories found in {self.artifacts_dir}",
            )
            return False

        print(f"Found {len(artifacts)} artifacts:")
        for artifact in artifacts:
            print(f"  - {artifact.name}")
        print()

        # Run checks
        self._check_manifests(artifacts)
        self._check_fat_binary_conversion(artifacts)
        self._check_architecture_separation(artifacts)
        self._check_kpack_archives(artifacts)

        # Print summary
        self._print_summary()

        return self.errors == 0

    def _discover_artifacts(self) -> list[Path]:
        """Discover artifact directories in artifacts directory."""
        if not self.artifacts_dir.exists():
            return []
        return [d for d in self.artifacts_dir.iterdir() if d.is_dir()]

    def _check_manifests(self, artifacts: list[Path]) -> None:
        """Verify all artifacts have valid manifest files."""
        print("CHECK: Artifact Manifests")
        print("-" * 70)

        details = []
        all_passed = True

        for artifact in artifacts:
            manifest_file = artifact / "artifact_manifest.txt"
            if not manifest_file.exists():
                details.append(f"  ✗ {artifact.name}: Missing artifact_manifest.txt")
                all_passed = False
            else:
                try:
                    with open(manifest_file) as f:
                        prefixes = [line.strip() for line in f if line.strip()]
                    details.append(f"  ✓ {artifact.name}: {len(prefixes)} prefixes")
                except Exception as e:
                    details.append(f"  ✗ {artifact.name}: Error reading manifest: {e}")
                    all_passed = False

        if all_passed:
            print("✓ All artifacts have valid manifests\n")
        else:
            print("✗ Manifest validation failed\n")
            self.errors += 1

        if self.verbose or not all_passed:
            for detail in details:
                print(detail)
            print()

        self.results.append(
            VerificationResult(
                "Artifact Manifests",
                all_passed,
                "All manifests present and valid"
                if all_passed
                else "Some manifests missing or invalid",
                details,
            )
        )

    def _check_fat_binary_conversion(self, artifacts: list[Path]) -> None:
        """Verify fat binaries were converted to host-only (PROGBITS -> NOBITS)."""
        print("CHECK: Fat Binary Conversion")
        print("-" * 70)

        details = []
        all_passed = True

        # Only check generic artifact
        generic_artifacts = [a for a in artifacts if "generic" in a.name]
        if not generic_artifacts:
            print("⊘ No generic artifact found, skipping fat binary check\n")
            self.results.append(
                VerificationResult(
                    "Fat Binary Conversion", True, "No generic artifact to check", []
                )
            )
            return

        for artifact in generic_artifacts:
            # Find all .so files
            so_files = list(artifact.glob("**/*.so*"))
            # Filter to actual files (not symlinks)
            so_files = [f for f in so_files if f.is_file() and not f.is_symlink()]

            converted_binaries = []
            host_only_binaries = []
            failed_binaries = []

            for so_file in so_files:
                file_size = so_file.stat().st_size
                size_mb = file_size / (1024 * 1024)
                rel_path = so_file.relative_to(artifact)

                # Check if has .hip_fatbin section
                section_info = self._get_hip_fatbin_section(so_file)
                if section_info is None:
                    host_only_binaries.append((rel_path, size_mb))
                    continue

                section_type, section_size = section_info

                if section_type == "PROGBITS":
                    failed_binaries.append(
                        (rel_path, size_mb, "Still has PROGBITS .hip_fatbin")
                    )
                    all_passed = False
                elif section_type == "NOBITS":
                    # Check for .rocm_kpack_ref marker
                    has_marker = self._has_kpack_ref(so_file)
                    if has_marker:
                        converted_binaries.append((rel_path, size_mb))
                    else:
                        failed_binaries.append(
                            (rel_path, size_mb, "NOBITS but missing .rocm_kpack_ref")
                        )
                        all_passed = False

            # Print summary
            details.append(
                f"  Summary: {len(converted_binaries)} converted, {len(host_only_binaries)} host-only, {len(failed_binaries)} failed"
            )
            details.append("")

            # Print converted binaries
            if converted_binaries:
                details.append("  Converted (fat → host-only):")
                for path, size in sorted(converted_binaries):
                    details.append(f"    ✓ {path} ({size:.1f}M)")
                details.append("")

            # Print host-only binaries
            if host_only_binaries:
                details.append("  Host-only (no device code):")
                for path, size in sorted(host_only_binaries):
                    details.append(f"    • {path} ({size:.1f}M)")
                details.append("")

            # Print failures
            if failed_binaries:
                details.append("  Failed conversions:")
                for path, size, reason in sorted(failed_binaries):
                    details.append(f"    ✗ {path} ({size:.1f}M) - {reason}")
                details.append("")

        if all_passed:
            print("✓ All fat binaries correctly converted to NOBITS\n")
        else:
            print("✗ Some binaries still have PROGBITS .hip_fatbin sections\n")
            self.errors += 1

        for detail in details:
            print(detail)
        print()

        self.results.append(
            VerificationResult(
                "Fat Binary Conversion",
                all_passed,
                "All fat binaries converted"
                if all_passed
                else "Some binaries not converted",
                details,
            )
        )

    def _check_architecture_separation(self, artifacts: list[Path]) -> None:
        """Verify architecture-specific artifacts only contain files for that architecture."""
        print("CHECK: Architecture Separation")
        print("-" * 70)

        details = []
        all_passed = True

        # Find arch-specific artifacts (not generic, not gfx906 which is minimal)
        arch_pattern = re.compile(r"gfx(\d+)")
        arch_artifacts = []
        for artifact in artifacts:
            match = arch_pattern.search(artifact.name)
            if match and "generic" not in artifact.name:
                arch_artifacts.append((artifact, match.group(0)))

        if not arch_artifacts:
            print("⊘ No architecture-specific artifacts found\n")
            self.results.append(
                VerificationResult(
                    "Architecture Separation",
                    True,
                    "No arch-specific artifacts to check",
                    [],
                )
            )
            return

        for artifact, expected_arch in arch_artifacts:
            # Find all files with gfx* in the name
            all_files = list(artifact.glob("**/*gfx*"))
            arch_files = [f for f in all_files if f.is_file()]

            contaminated = []
            for file in arch_files:
                # Extract all gfx architectures mentioned in filename
                found_archs = arch_pattern.findall(file.name)
                for found_arch in found_archs:
                    if f"gfx{found_arch}" != expected_arch:
                        contaminated.append((file, f"gfx{found_arch}"))

            if contaminated:
                details.append(
                    f"  ✗ {artifact.name}: {len(contaminated)} files from other architectures"
                )
                for file, wrong_arch in contaminated[:5]:  # Show first 5
                    details.append(
                        f"      - {file.relative_to(artifact)} contains {wrong_arch}"
                    )
                if len(contaminated) > 5:
                    details.append(f"      ... and {len(contaminated) - 5} more")
                all_passed = False
            else:
                details.append(
                    f"  ✓ {artifact.name}: All files are {expected_arch} (checked {len(arch_files)} files)"
                )

        if all_passed:
            print("✓ All architectures correctly separated\n")
        else:
            print("✗ Found architecture cross-contamination\n")
            self.errors += 1

        for detail in details:
            print(detail)
        print()

        self.results.append(
            VerificationResult(
                "Architecture Separation",
                all_passed,
                "All archs separated"
                if all_passed
                else "Architecture cross-contamination detected",
                details,
            )
        )

    def _check_kpack_archives(self, artifacts: list[Path]) -> None:
        """Verify kpack archives are present and valid."""
        print("CHECK: Kpack Archives")
        print("-" * 70)

        details = []
        all_passed = True

        # Find artifacts with kpack directories
        for artifact in artifacts:
            kpack_dir = artifact / "kpack" / "stage" / ".kpack"
            if not kpack_dir.exists():
                continue

            kpack_files = list(kpack_dir.glob("*.kpack"))
            if not kpack_files:
                details.append(
                    f"  ✗ {artifact.name}: kpack directory exists but no .kpack files"
                )
                all_passed = False
                continue

            for kpack_file in kpack_files:
                # Check file exists and has content
                if kpack_file.stat().st_size == 0:
                    details.append(f"  ✗ {artifact.name}: {kpack_file.name} is empty")
                    all_passed = False
                    continue

                # Try to read kpack structure
                try:
                    with open(kpack_file, "rb") as f:
                        # Read binary header
                        magic = f.read(4)
                        if magic != b"KPAK":
                            details.append(
                                f"  ✗ {artifact.name}: {kpack_file.name} has invalid magic: {magic}"
                            )
                            all_passed = False
                            continue

                        version = int.from_bytes(f.read(4), byteorder="little")
                        toc_offset = int.from_bytes(f.read(8), byteorder="little")

                        # Seek to TOC and read MessagePack
                        f.seek(toc_offset)
                        unpacker = msgpack.Unpacker(f, raw=False)
                        toc = next(unpacker)

                    if not isinstance(toc, dict):
                        details.append(
                            f"  ✗ {artifact.name}: {kpack_file.name} has invalid TOC"
                        )
                        all_passed = False
                        continue

                    # Count kernels in TOC
                    kernel_count = 0
                    toc_dict = toc.get("toc", {})
                    for binary_path, archs in toc_dict.items():
                        for arch, kernel_info in archs.items():
                            kernel_count += 1

                    size_mb = kpack_file.stat().st_size / (1024 * 1024)
                    details.append(
                        f"  ✓ {artifact.name}: {kpack_file.name} ({size_mb:.1f}MB, {kernel_count} kernels)"
                    )

                except Exception as e:
                    details.append(
                        f"  ✗ {artifact.name}: {kpack_file.name} invalid: {e}"
                    )
                    all_passed = False

        if not details:
            print("⊘ No kpack archives found\n")
            self.results.append(
                VerificationResult(
                    "Kpack Archives", True, "No kpack archives to check", []
                )
            )
            return

        if all_passed:
            print("✓ All kpack archives are valid\n")
        else:
            print("✗ Some kpack archives are invalid\n")
            self.errors += 1

        for detail in details:
            print(detail)
        print()

        self.results.append(
            VerificationResult(
                "Kpack Archives",
                all_passed,
                "All kpack archives valid"
                if all_passed
                else "Some kpack archives invalid",
                details,
            )
        )

    def _get_hip_fatbin_section(self, binary_path: Path) -> Optional[tuple[str, int]]:
        """Get .hip_fatbin section type and size. Returns None if no section."""
        section_type = get_section_type(
            binary_path, ".hip_fatbin", toolchain=self.toolchain
        )
        if section_type is None:
            return None
        # Size is not critical for verification, just need to know type
        return (section_type, 0)

    def _has_kpack_ref(self, binary_path: Path) -> bool:
        """Check if binary has .rocm_kpack_ref section."""
        return has_section(binary_path, ".rocm_kpack_ref", toolchain=self.toolchain)

    def _fail(self, check_name: str, message: str) -> None:
        """Record a failed check."""
        self.results.append(VerificationResult(check_name, False, message, []))
        self.errors += 1

    def _print_summary(self) -> None:
        """Print verification summary."""
        print("=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        for result in self.results:
            status = "✓" if result.passed else "✗"
            print(f"{status} {result.check_name}: {result.message}")

        print()
        print(f"Total checks: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print()

        if failed == 0:
            print("✓ ALL CHECKS PASSED")
            print()
        else:
            print("✗ VERIFICATION FAILED")
            print(f"  {failed} check(s) failed")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify split artifacts meet expected invariants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool validates that artifact splitting produced correct output:
  - Fat binaries converted to host-only (PROGBITS -> NOBITS)
  - Architecture separation is correct
  - Kpack archives are valid
  - Manifest files are present

Exit codes:
  0 - All checks passed
  1 - One or more checks failed
  2 - Verification could not run (invalid arguments, etc.)

Examples:
  # Verify split artifacts
  %(prog)s --artifacts-dir /path/to/split/artifacts

  # Verbose mode
  %(prog)s --artifacts-dir /path/to/split/artifacts --verbose
""",
    )

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing split artifacts to verify",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--clang-offload-bundler",
        type=Path,
        default=None,
        help="Path to clang-offload-bundler tool (optional, for toolchain initialization)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.artifacts_dir.exists():
        print(
            f"Error: Artifacts directory does not exist: {args.artifacts_dir}",
            file=sys.stderr,
        )
        return 2

    if not args.artifacts_dir.is_dir():
        print(f"Error: Path is not a directory: {args.artifacts_dir}", file=sys.stderr)
        return 2

    # Create toolchain
    try:
        toolchain = Toolchain(clang_offload_bundler=args.clang_offload_bundler)
    except Exception as e:
        print(f"Error: Failed to initialize toolchain: {e}", file=sys.stderr)
        return 2

    # Run verification
    verifier = ArtifactVerifier(args.artifacts_dir, toolchain, verbose=args.verbose)
    success = verifier.run_all_checks()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
