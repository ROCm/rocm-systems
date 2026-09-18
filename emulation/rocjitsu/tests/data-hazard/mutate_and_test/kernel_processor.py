# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Optional, List, Tuple


@dataclass
class KernelBuilderConfig:
    tools: dict
    shader_cpp: Path
    asm_file: Path
    arch: str
    workdir: Path
    tag: str
    rocm_path: str = ""
    target_features: Optional[List[str]] = None

    @property
    def host_obj_path(self) -> Path:
        return self.workdir / f"{self.shader_cpp.stem}_host.o"

    @property
    def dev_obj_path(self) -> Path:
        return self.workdir / f"{self.shader_cpp.stem}_dev.o"

    @property
    def bundle_path(self) -> Path:
        return self.workdir / f"{self.shader_cpp.stem}_hipfb"

    @property
    def fatbin_obj_path(self) -> Path:
        return self.workdir / f"{self.shader_cpp.stem}_fatbin.o"

    @property
    def exe_path(self) -> Path:
        return self.workdir / f"{self.shader_cpp.stem}"

    @property
    def baseline_hazard_report_path(self) -> Path:
        return (
            self.workdir
            / f"{self.shader_cpp.stem}_reports"
            / f"data_hazard_report_{self.shader_cpp.stem}.json"
        )

    def mutant_hazard_report_path(self, tag: str) -> Path:
        return (
            self.workdir
            / f"{self.shader_cpp.stem}_reports"
            / f"data_hazard_report_{self.shader_cpp.stem}_{tag}.json"
        )


def run_cmd(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command, returning the CompletedProcess."""
    if verbose:
        import sys

        print(f"  [CMD] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)


class KernelBuilder:
    def __init__(self, config: KernelBuilderConfig):
        self._config = config

    @property
    def config(self) -> KernelBuilderConfig:
        return self._config

    def build_from_asm(self) -> Tuple[Optional[Path], str]:
        """
        Full build pipeline: assembly → device obj → bundle → fatbin → link.

        Returns (exe_path, error_string).  exe_path is None on failure.
        """
        config = self._config
        tools = config.tools

        required_tools = ["hipcc", "clang", "bundler", "llvm_nm", "llvm_mc", "hipcc"]
        for tool in required_tools:
            if tool not in tools:
                return None, f"tool: {tool} not found"

        # Host object — recompile every time to ensure the HIP runtime's
        # embedded device code hash matches the current device binary.
        host_obj = config.host_obj_path
        ok, err = self._compile_host_object(config)
        if not ok:
            return None, f"host compile: {err}"

        # Device object
        ok, err = self._assemble_device(config)
        if not ok:
            return None, f"device assemble: {err}"

        # Offload bundle
        ok, err = self._create_offload_bundle(config)
        if not ok:
            return None, f"bundle: {err}"

        # Extract the fatbin symbol name from the host object
        fatbin_sym = "__hip_fatbin"
        if "llvm_nm" in tools:
            sym = self._extract_fatbin_symbol(config)
            if sym:
                fatbin_sym = sym
        ok, err = self._embed_fatbin(config, fatbin_sym)
        if not ok:
            return None, f"embed: {err}"

        # Link
        ok, err = self._link_executable(config)
        if not ok:
            return None, f"link: {err}"

        return config.exe_path, ""

    def compile_to_assembly(
        self, extra_cxxflags: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """Compile a HIP shader to device-only assembly."""
        config = self.config
        hipcc = config.tools["hipcc"]
        arch = config.arch
        shader = config.shader_cpp
        output = config.asm_file
        rocm_path = config.rocm_path
        target_features = config.target_features

        cmd = [
            hipcc,
            "-S",
            "--cuda-device-only",
            f"--offload-arch={arch}",
            str(shader),
            "-o",
            str(output),
        ]
        if rocm_path:
            cmd.insert(1, f"--rocm-path={rocm_path}")
            inc = Path(rocm_path) / "include"
            if inc.is_dir():
                cmd.extend(["-I", str(inc)])

        for feat in target_features or []:
            cmd.extend(["-Xclang", "-target-feature", "-Xclang", f"+{feat}"])

        cmd.extend(extra_cxxflags or [])
        # hipcc may ignore -o with --cuda-device-only and emit an auto-named .s
        auto_name = f"{shader.stem}-hip-amdgcn-amd-amdhsa-{arch}.s"
        auto_path = Path.cwd() / auto_name

        for stale in (output, auto_path):
            stale.unlink(missing_ok=True)

        r = run_cmd(cmd)
        if r.returncode != 0:
            return False, r.stderr

        # Prefer the explicitly requested output if it exists
        if output.is_file():
            return True, ""
        # Fall back to the auto-named file if hipcc ignored -o
        if auto_path.is_file():
            shutil.move(str(auto_path), str(output))
            return True, ""
        # Neither expected output was produced; report failure
        return False, r.stderr

    def _compile_host_object(self, config: KernelBuilderConfig) -> Tuple[bool, str]:
        """Compile the host-only object from a HIP source."""
        hipcc = config.tools["hipcc"]
        shader = config.shader_cpp
        output = config.host_obj_path
        rocm_path = config.rocm_path
        cmd = [hipcc, "-c", "--cuda-host-only", str(shader), "-o", str(output)]
        if rocm_path:
            cmd.insert(1, f"--rocm-path={rocm_path}")
            inc = Path(rocm_path) / "include"
            if inc.is_dir():
                cmd.extend(["-I", str(inc)])

        result = run_cmd(cmd)
        return result.returncode == 0, result.stderr

    def _assemble_device(self, config: KernelBuilderConfig) -> Tuple[bool, str]:
        """Assemble device .s to .o."""
        clang = config.tools["clang"]
        asm_file = config.asm_file
        arch = config.arch
        output = config.dev_obj_path
        mattr = ",".join(f"+{f}" for f in (config.target_features or []))
        cmd = [
            clang,
            "-target",
            "amdgcn-amd-amdhsa",
            f"-mcpu={arch}",
            str(asm_file),
            "-o",
            str(output),
        ]
        if mattr:
            cmd.insert(4, f"-mattr={mattr}")
        r = run_cmd(cmd)
        return r.returncode == 0, r.stderr

    def _create_offload_bundle(self, config: KernelBuilderConfig) -> Tuple[bool, str]:
        """Bundle device object into a HIP fat binary."""
        bundler = config.tools["bundler"]
        device_obj = config.dev_obj_path
        arch = config.arch
        output = config.bundle_path

        targets = f"host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--{arch}"
        cmd = [
            bundler,
            "-type=o",
            "-bundle-align=4096",
            f"-targets={targets}",
            "-input=/dev/null",
            f"-input={str(device_obj)}",
            f"-output={str(output)}",
        ]
        r = run_cmd(cmd)
        return r.returncode == 0, r.stderr

    def _extract_fatbin_symbol(self, config: KernelBuilderConfig) -> Optional[str]:
        """Extract the __hip_fatbin_* symbol name expected by the host object."""
        llvm_nm = config.tools["llvm_nm"]
        host_obj = config.host_obj_path

        cmd = [llvm_nm, str(host_obj)]
        r = run_cmd(cmd)

        if r.returncode != 0:
            # try system nm as fallback
            r = run_cmd(["nm", str(host_obj)])
        for line in r.stdout.splitlines():
            m = re.search(r"\b(__hip_fatbin\w*)\b", line)
            if m and "wrapper" not in m.group(1):
                return m.group(1)
        return None

    def _embed_fatbin(
        self,
        config: KernelBuilderConfig,
        fatbin_symbol: str = "__hip_fatbin",
    ) -> Tuple[bool, str]:
        """Create a host object that embeds the offload bundle via .incbin."""
        mcin = config.workdir / "hip_obj_gen.mcin"
        mcin.write_text(
            f"    .type {fatbin_symbol},@object\n"
            '    .section .hip_fatbin,"a",@progbits\n'
            f"    .globl {fatbin_symbol}\n"
            "    .p2align 12\n"
            f"{fatbin_symbol}:\n"
            f'    .incbin "{str(config.bundle_path)}"\n'
        )
        cmd = [
            config.tools["llvm_mc"],
            "-triple",
            "x86_64-unknown-linux-gnu",
            "-o",
            str(config.fatbin_obj_path),
            str(mcin),
            "--filetype=obj",
        ]
        r = run_cmd(cmd)
        return r.returncode == 0, r.stderr

    def _link_executable(
        self,
        config: KernelBuilderConfig,
    ) -> Tuple[bool, str]:
        """Link host + fatbin objects into the final executable."""
        # Use hipcc --no-offload-new-driver to prevent hipcc from
        # re-processing or recompiling device code during linking.
        cmd = [
            config.tools["hipcc"],
            "--no-offload-new-driver",
            "-o",
            str(config.exe_path),
            str(config.host_obj_path),
            str(config.fatbin_obj_path),
        ]
        rocm_path = config.rocm_path
        if rocm_path:
            cmd.extend(
                [
                    f"--rocm-path={rocm_path}",
                    f"-L{rocm_path}/lib",
                    f"-Wl,-rpath,{rocm_path}/lib",
                ]
            )
        r = run_cmd(cmd)
        if r.returncode != 0:
            # Fallback without --no-offload-new-driver
            cmd = [
                config.tools["hipcc"],
                "-o",
                str(config.exe_path),
                str(config.host_obj_path),
                str(config.fatbin_obj_path),
            ]
            if rocm_path:
                cmd.extend(
                    [
                        f"--rocm-path={rocm_path}",
                        f"-L{rocm_path}/lib",
                        f"-Wl,-rpath,{rocm_path}/lib",
                    ]
                )
            r = run_cmd(cmd)
        return r.returncode == 0, r.stderr
