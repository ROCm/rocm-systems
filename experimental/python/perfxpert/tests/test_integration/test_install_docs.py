"""Guard the documented Git install path and cross-distro Python guidance."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[2]
_README = _APP_ROOT / "README.md"
_GETTING_STARTED = _APP_ROOT / "docs" / "guides" / "getting-started.md"
_AGENTIC_MODE = _APP_ROOT / "docs" / "guides" / "agentic-mode.md"
_INSTALL_WRAPPER = _APP_ROOT / "scripts" / "pip-install-from-git.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _env_with_path(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(path)
    env.pop("PERFXPERT_AUTO_INSTALL_PREREQS", None)
    env.pop("PERFXPERT_AUTO_INSTALL_BUN", None)
    return env


def _symlink_tool(bin_dir: Path, name: str) -> None:
    tool = shutil.which(name)
    assert tool is not None, f"Missing host tool needed by test: {name}"
    os.symlink(tool, bin_dir / name)


def _write_fake_perfxpert_code(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "perfxpert-code",
        "#!/bin/sh\necho 'AMD ROCm PerfXpert 0.2.0 (opencode wrapper)'\n",
    )


def test_install_wrapper_exists_and_is_non_empty() -> None:
    assert _INSTALL_WRAPPER.is_file(), f"Missing install wrapper: {_INSTALL_WRAPPER}"
    assert _INSTALL_WRAPPER.stat().st_size > 1_000, (
        "Install wrapper is unexpectedly tiny; likely truncated."
    )


def test_install_wrapper_help_mentions_supported_prereqs() -> None:
    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "curl" in result.stdout
    assert "unzip" in result.stdout
    assert "python3 -m venv .venv" in result.stdout
    assert "python3-venv" in result.stdout
    assert "python3-pip" in result.stdout
    assert "git" in result.stdout
    assert "PERFXPERT_AUTO_INSTALL_BUN=1" in result.stdout
    assert "PERFXPERT_AUTO_INSTALL_PREREQS=1" in result.stdout
    assert "It never downloads" in result.stdout
    assert "bun --version" in result.stdout
    assert "SLES 15.6" in result.stdout
    assert "command -v curl >/dev/null || dnf install -y curl" in result.stdout
    assert "dnf install -y git unzip python3.11 python3.11-pip" in result.stdout
    assert "dnf install -y git unzip python3 python3-pip" in result.stdout
    assert "zypper install -y curl git unzip python311 python311-pip" in result.stdout
    assert "python3.11 -m venv .venv" in result.stdout
    assert 'REF=<SHA>; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"' in result.stdout


def test_install_wrapper_help_works_from_stdin() -> None:
    result = subprocess.run(
        ["/bin/bash", "-lc", f"/bin/bash -s -- --help < {shlex.quote(str(_INSTALL_WRAPPER))}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "pip-install-from-git.sh" in result.stdout
    assert "curl -fsSL" in result.stdout
    assert "PERFXPERT_AUTO_INSTALL_BUN=1" in result.stdout
    assert "PERFXPERT_AUTO_INSTALL_PREREQS=1" in result.stdout
    assert "It never downloads" in result.stdout
    assert "bun --version" in result.stdout
    assert "SLES 15.6" in result.stdout
    assert "command -v curl >/dev/null || dnf install -y curl" in result.stdout
    assert "dnf install -y git unzip python3.11 python3.11-pip" in result.stdout
    assert "zypper install -y curl git unzip python311 python311-pip" in result.stdout
    assert 'REF=<SHA>; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"' in result.stdout


def test_readme_keeps_customer_install_flow_curl_only() -> None:
    text = _README.read_text(encoding="utf-8")
    assert "curl" in text
    assert "unzip" in text
    assert "python3 -m venv .venv" in text
    assert "python3-venv" in text
    assert "python3-pip" in text
    assert "PERFXPERT_AUTO_INSTALL_BUN=1" not in text
    assert "No separate `opencode` install is needed" in text
    assert "patched bundled `perfxpert-code` binary" in text
    assert "builds the patched bundled" in text
    assert "Ubuntu/RHEL/SLES package matrix" in text
    assert "| `private` | Any OpenAI-compatible endpoint |" in text
    assert "PERFXPERT_LLM_PRIVATE_URL" in text
    assert "PERFXPERT_LLM_PRIVATE_MODEL" in text
    assert "PERFXPERT_LLM_PRIVATE_API_KEY" in text
    assert "PERFXPERT_LLM_PRIVATE_HEADERS" in text
    assert "https://llm-api.iexample.com/OpenAI" in text
    assert "Ocp-Apim-Subscription-Key" in text
    assert 'REF=<SHA>; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"' in text
    assert "GIT_CONFIG_COUNT=1" not in text
    assert "git clone --depth 1 --no-recurse-submodules" not in text
    assert "wget -qO-" not in text
    assert "curl -fsSL https://bun.sh/install | bash" not in text


def test_readme_keeps_copy_paste_command_shapes() -> None:
    text = _README.read_text(encoding="utf-8")
    assert "perfxpert analyze -i trace.db --llm anthropic --format webview -o report.html" in text
    assert "perfxpert analyze -i trace.db --llm openai --llm-model gpt-4o-mini" in text
    assert "PERFXPERT_AIRGAP=1 perfxpert analyze -i trace.db" in text
    assert "\nperfxpert-code\n" in text
    assert "perfxpert-code claude" in text
    assert "perfxpert-code codex" in text
    assert "perfxpert-code gemini" in text
    assert 'PERFXPERT_OPENCODE_PATH="$(command -v opencode)" perfxpert-code opencode' in text


def test_getting_started_keeps_internal_install_detail() -> None:
    text = _GETTING_STARTED.read_text(encoding="utf-8")
    assert "curl" in text
    assert "unzip" in text
    assert "python3 -m venv .venv" in text
    assert "python3-venv" in text
    assert "python3-pip" in text
    assert "never downloads a separate" in text
    assert "Python runtime" in text
    assert "PERFXPERT_AUTO_INSTALL_BUN=1" in text
    for distro in (
        "Ubuntu 22.04",
        "Ubuntu 24.04",
        "UBI/RHEL 9",
        "UBI/RHEL 10",
        "SLES 15.6",
    ):
        assert distro in text
    assert "curl-minimal" in text
    assert "command -v curl >/dev/null || dnf install -y curl" in text
    assert "dnf install -y git unzip python3.11 python3.11-pip" in text
    assert "dnf install -y git unzip python3 python3-pip" in text
    assert "zypper install -y curl git unzip python311 python311-pip" in text
    assert "bun run build --single\n--skip-install" in text
    assert "PERFXPERT_LLM_PRIVATE_API_KEY` or `--llm-api-key" in text
    assert "https://llm-api.iexample.com/OpenAI" in text
    assert "Ocp-Apim-Subscription-Key" in text
    assert "python3.11 -m venv .venv" in text
    assert 'REF=<SHA>; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"' in text
    assert "GIT_CONFIG_COUNT=1" in text
    assert "bash rocm-systems/experimental/python/perfxpert/scripts/pip-install-from-git.sh <SHA>" in text
    assert "wget -qO-" not in text


def test_install_docs_do_not_recommend_break_system_packages() -> None:
    for doc in (_README, _GETTING_STARTED):
        text = doc.read_text(encoding="utf-8")
        assert "--break-system-packages" not in text, (
            f"{doc} must not recommend --break-system-packages."
        )


def test_install_docs_explain_perfxpert_code_follow_up() -> None:
    readme = _README.read_text(encoding="utf-8")
    guide = _GETTING_STARTED.read_text(encoding="utf-8")

    assert "PERFXPERT_AUTO_INSTALL_BUN=1" not in readme
    assert "verifies `perfxpert-code`" in readme
    assert "curl -fsSL https://bun.sh/install | bash" not in readme
    assert "Direct\npip/editable paths use the same `setup.py` build hook" in guide
    assert "pip fails with an actionable prerequisite" in guide
    assert "opencode.ai/install" not in readme


def test_provider_docs_keep_private_endpoint_contract_current() -> None:
    for doc in (_README, _GETTING_STARTED, _AGENTIC_MODE):
        text = doc.read_text(encoding="utf-8")
        assert "PERFXPERT_LLM_PRIVATE_URL" in text
        assert "PERFXPERT_LLM_PRIVATE_MODEL" in text
        assert "PERFXPERT_LLM_PRIVATE_API_KEY" in text
        assert "PERFXPERT_LLM_PRIVATE_HEADERS" in text
        assert "https://llm-api.iexample.com/OpenAI" in text
        assert "Ocp-Apim-Subscription-Key" in text
        assert "ROCPD_LLM_PRIVATE" not in text


def test_install_wrapper_prints_missing_prereqs_without_auto_install(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    os.symlink(shutil.which("python3"), bin_dir / "python3")
    _symlink_tool(bin_dir, "cat")
    _symlink_tool(bin_dir, "curl")

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER)],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "missing OS prerequisites" in result.stderr
    assert "refusing to auto-install prerequisites by default" in result.stderr
    assert "PERFXPERT_AUTO_INSTALL_PREREQS=1" in result.stderr
    assert "git" in result.stderr
    assert "apt install -y curl git unzip python3-venv python3-pip" in result.stderr
    assert "command -v curl >/dev/null || dnf install -y curl" in result.stderr
    assert "dnf install -y git unzip python3.11 python3.11-pip" in result.stderr
    assert "dnf install -y git unzip python3 python3-pip" in result.stderr
    assert "zypper install -y curl git unzip python311 python311-pip" in result.stderr


def test_install_wrapper_does_not_run_package_manager_without_opt_in(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    os.symlink(shutil.which("python3"), bin_dir / "python3")
    _symlink_tool(bin_dir, "cat")
    marker = tmp_path / "apt-ran.txt"
    _write_executable(
        bin_dir / "apt-get",
        f"""#!/bin/bash
echo "$*" >> "{marker}"
exit 0
""",
    )

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER)],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "refusing to auto-install prerequisites by default" in result.stderr
    assert not marker.exists()


def test_install_wrapper_double_dash_does_not_enable_prereq_auto_install(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    os.symlink(shutil.which("python3"), bin_dir / "python3")
    _symlink_tool(bin_dir, "cat")
    marker = tmp_path / "apt-ran.txt"
    _write_executable(
        bin_dir / "apt-get",
        f"""#!/bin/bash
echo "$*" >> "{marker}"
exit 0
""",
    )

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "--", "--auto-install-prereqs"],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "refusing to auto-install prerequisites by default" in result.stderr
    assert not marker.exists()


def test_install_wrapper_auto_install_reports_missing_package_manager(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    os.symlink(shutil.which("python3"), bin_dir / "python3")
    _symlink_tool(bin_dir, "cat")
    _symlink_tool(bin_dir, "curl")

    env = _env_with_path(bin_dir)
    env["PERFXPERT_AUTO_INSTALL_PREREQS"] = "1"
    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "missing OS prerequisites" in result.stderr
    assert "no supported package manager found" in result.stderr


def test_install_wrapper_auto_installs_missing_prereqs_with_apt_get(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bundled = tmp_path / "site-packages" / "perfxpert" / "_bundled" / "opencode"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("fake-opencode", encoding="utf-8")
    bundled.chmod(0o755)
    _write_executable(
        bin_dir / "python3",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 1
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
    *'resources.files("perfxpert")'*)
      echo "{bundled}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "cat")
    _symlink_tool(bin_dir, "chmod")
    _symlink_tool(bin_dir, "git")
    _write_fake_perfxpert_code(bin_dir)
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
"$@"
""",
    )
    installed = tmp_path / "installed.txt"
    _write_executable(
        bin_dir / "apt-get",
        f"""#!/bin/bash
echo "$*" >> "{installed}"
if [ "$1" = "install" ]; then
  printf '#!/bin/sh\\nexit 0\\n' > "{bin_dir}/curl"
  printf '#!/bin/sh\\nexit 0\\n' > "{bin_dir}/unzip"
  chmod +x "{bin_dir}/curl" "{bin_dir}/unzip"
fi
exit 0
""",
    )

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "--auto-install-prereqs", "develop"],
        capture_output=True,
        text=True,
        env={**_env_with_path(bin_dir), "PERFXPERT_AUTO_INSTALL_BUN": "1"},
        check=False,
    )

    assert result.returncode == 0
    assert installed.read_text(encoding="utf-8").splitlines() == [
        "update",
        "install -y curl unzip",
    ]


def test_install_wrapper_dnf_keeps_existing_curl_provider(tmp_path: Path) -> None:
    """RHEL/UBI images can provide `curl` through curl-minimal.

    The wrapper must not ask dnf to install full `curl` again when the command
    is already present, because that conflicts with curl-minimal on UBI.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bundled = tmp_path / "site-packages" / "perfxpert" / "_bundled" / "opencode"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("fake-opencode", encoding="utf-8")
    bundled.chmod(0o755)
    _write_executable(
        bin_dir / "python3",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 1
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
    *'resources.files("perfxpert")'*)
      echo "{bundled}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "cat")
    _symlink_tool(bin_dir, "chmod")
    _write_executable(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
    _write_fake_perfxpert_code(bin_dir)
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
"$@"
""",
    )
    installed = tmp_path / "installed.txt"
    _write_executable(
        bin_dir / "dnf",
        f"""#!/bin/bash
echo "$*" >> "{installed}"
if [ "$1" = "install" ]; then
  printf '#!/bin/sh\\nexit 0\\n' > "{bin_dir}/git"
  printf '#!/bin/sh\\nexit 0\\n' > "{bin_dir}/unzip"
  /bin/chmod +x "{bin_dir}/git" "{bin_dir}/unzip"
fi
exit 0
""",
    )

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env={**_env_with_path(bin_dir), "PERFXPERT_AUTO_INSTALL_PREREQS": "1"},
        check=False,
    )

    assert result.returncode == 0
    assert installed.read_text(encoding="utf-8").splitlines() == [
        "install -y git",
    ]


def test_install_wrapper_auto_installs_missing_prereqs_with_zypper(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bundled = tmp_path / "site-packages" / "perfxpert" / "_bundled" / "opencode"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("fake-opencode", encoding="utf-8")
    bundled.chmod(0o755)
    _write_executable(
        bin_dir / "python3",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 1
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
    *'resources.files("perfxpert")'*)
      echo "{bundled}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "cat")
    _symlink_tool(bin_dir, "chmod")
    _write_executable(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
    _write_fake_perfxpert_code(bin_dir)
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
"$@"
""",
    )
    installed = tmp_path / "installed.txt"
    _write_executable(
        bin_dir / "zypper",
        f"""#!/bin/bash
echo "$*" >> "{installed}"
if [ "$1" = "--non-interactive" ] && [ "$2" = "install" ]; then
  printf '#!/bin/sh\\nexit 0\\n' > "{bin_dir}/git"
  printf '#!/bin/sh\\nexit 0\\n' > "{bin_dir}/unzip"
  /bin/chmod +x "{bin_dir}/git" "{bin_dir}/unzip"
fi
exit 0
""",
    )

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env={**_env_with_path(bin_dir), "PERFXPERT_AUTO_INSTALL_PREREQS": "1"},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert installed.read_text(encoding="utf-8").splitlines() == [
        "--non-interactive install -y git",
    ]


def test_install_wrapper_rejects_python_older_than_310(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "python3",
        """#!/bin/bash
if [ "$1" = "-c" ]; then
  exit 1
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "curl")
    _symlink_tool(bin_dir, "unzip")
    _symlink_tool(bin_dir, "git")
    _symlink_tool(bin_dir, "cat")

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER)],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "missing OS prerequisites" in result.stderr
    assert "python3.10+ with pip" in result.stderr
    assert "refusing to auto-install prerequisites by default" in result.stderr


def test_install_wrapper_fails_when_python_m_pip_is_missing(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "python3",
        """#!/bin/bash
if [ "$1" = "-c" ]; then
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  exit 1
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "curl")
    _symlink_tool(bin_dir, "unzip")
    _symlink_tool(bin_dir, "git")
    _symlink_tool(bin_dir, "cat")

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER)],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "missing OS prerequisites" in result.stderr
    assert "python3.10+ with pip" in result.stderr
    assert "refusing to auto-install prerequisites by default" in result.stderr
    assert "apt install -y curl git unzip python3-venv python3-pip" in result.stderr
    assert "command -v curl >/dev/null || dnf install -y curl" in result.stderr
    assert "dnf install -y git unzip python3.11 python3.11-pip" in result.stderr
    assert "dnf install -y git unzip python3 python3-pip" in result.stderr
    assert "zypper install -y curl git unzip python311 python311-pip" in result.stderr


def test_install_wrapper_skips_active_python_without_pip_for_versioned_fallback(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    chosen = tmp_path / "chosen.txt"
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    bundled = tmp_path / "site-packages" / "perfxpert" / "_bundled" / "opencode"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("fake-opencode", encoding="utf-8")
    bundled.chmod(0o755)

    _write_executable(
        bin_dir / "python",
        """#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  exit 1
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _write_executable(
        bin_dir / "python3.11",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
  echo "python3.11" > "{chosen}"
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 1
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
    *'resources.files("perfxpert")'*)
      echo "{bundled}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "curl")
    _symlink_tool(bin_dir, "unzip")
    _symlink_tool(bin_dir, "git")
    _symlink_tool(bin_dir, "cat")
    _write_fake_perfxpert_code(bin_dir)

    env = _env_with_path(bin_dir)
    env["HOME"] = str(home_dir)
    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert chosen.read_text(encoding="utf-8").strip() == "python3.11"


def test_install_wrapper_rejects_externally_managed_python(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    managed = tmp_path / "stdlib" / "EXTERNALLY-MANAGED"
    managed.parent.mkdir()
    managed.write_text("", encoding="utf-8")
    _write_executable(
        bin_dir / "python3",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 0
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{managed}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "curl")
    _symlink_tool(bin_dir, "unzip")
    _symlink_tool(bin_dir, "cat")
    _symlink_tool(bin_dir, "git")

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER)],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "the current Python is externally managed" in result.stderr
    assert "curl -fsSL" in result.stderr
    assert "command -v curl >/dev/null || dnf install -y curl" in result.stderr
    assert "dnf install -y git unzip python3.11 python3.11-pip" in result.stderr
    assert "dnf install -y git unzip python3 python3-pip" in result.stderr
    assert "zypper install -y curl git unzip python311 python311-pip" in result.stderr


def test_install_wrapper_reports_ready_bundle_from_pip_build(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    version_marker = tmp_path / "perfxpert-code-version.txt"
    bundled = tmp_path / "site-packages" / "perfxpert" / "_bundled" / "opencode"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("fake-opencode", encoding="utf-8")
    bundled.chmod(0o755)
    _write_executable(
        bin_dir / "python3",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
  cat > "{bin_dir}/perfxpert-code" <<'SH'
#!/bin/sh
echo "$*" >> "{version_marker}"
echo 'AMD ROCm PerfXpert 0.2.0 (opencode wrapper)'
SH
  /bin/chmod +x "{bin_dir}/perfxpert-code"
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 1
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
    *'resources.files("perfxpert")'*)
      echo "{bundled}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "curl")
    _symlink_tool(bin_dir, "unzip")
    _symlink_tool(bin_dir, "git")
    _symlink_tool(bin_dir, "cat")

    env = _env_with_path(bin_dir)
    env["HOME"] = str(home_dir)
    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "perfxpert-code ready at" in result.stderr
    assert "bundled patched opencode ready at" in result.stderr
    assert version_marker.read_text(encoding="utf-8").strip() == "--version"


def test_install_wrapper_fails_when_perfxpert_code_entrypoint_missing(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    bundled = tmp_path / "site-packages" / "perfxpert" / "_bundled" / "opencode"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("fake-opencode", encoding="utf-8")
    bundled.chmod(0o755)
    _write_executable(
        bin_dir / "python3",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 1
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
    *'resources.files("perfxpert")'*)
      echo "{bundled}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "curl")
    _symlink_tool(bin_dir, "unzip")
    _symlink_tool(bin_dir, "git")
    _symlink_tool(bin_dir, "cat")

    env = _env_with_path(bin_dir)
    env["HOME"] = str(home_dir)
    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "perfxpert-code` console entry point is not usable" in result.stderr
    assert "missing executable perfxpert-code console script" in result.stderr


def test_install_wrapper_fails_when_bundled_opencode_is_missing_after_install(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    home_dir = tmp_path / "home"
    home_dir.mkdir()

    _write_executable(
        bin_dir / "python3",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 1
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
    *'resources.files("perfxpert")'*)
      exit 1
      ;;
  esac
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "curl")
    _symlink_tool(bin_dir, "unzip")
    _symlink_tool(bin_dir, "git")
    _symlink_tool(bin_dir, "cat")
    _write_fake_perfxpert_code(bin_dir)

    env = _env_with_path(bin_dir)
    env["HOME"] = str(home_dir)
    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "bundled patched opencode binary was not produced" in result.stderr


def test_install_wrapper_prefers_active_python_before_versioned_fallbacks(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    chosen = tmp_path / "chosen.txt"
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    bundled = tmp_path / "site-packages" / "perfxpert" / "_bundled" / "opencode"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("fake-opencode", encoding="utf-8")
    bundled.chmod(0o755)

    _write_executable(
        bin_dir / "python",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
  echo "python" > "{chosen}"
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 1
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
    *'resources.files("perfxpert")'*)
      echo "{bundled}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _write_executable(
        bin_dir / "python3.11",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
  echo "python3.11" > "{chosen}"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 1
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
    *'resources.files("perfxpert")'*)
      echo "{bundled}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "curl")
    _symlink_tool(bin_dir, "unzip")
    _symlink_tool(bin_dir, "git")
    _symlink_tool(bin_dir, "cat")
    _write_fake_perfxpert_code(bin_dir)

    env = _env_with_path(bin_dir)
    env["HOME"] = str(home_dir)
    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert chosen.read_text(encoding="utf-8").strip() == "python"


def test_install_wrapper_falls_back_to_supported_versioned_python(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    chosen = tmp_path / "chosen.txt"
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    bundled = tmp_path / "site-packages" / "perfxpert" / "_bundled" / "opencode"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("fake-opencode", encoding="utf-8")
    bundled.chmod(0o755)

    _write_executable(
        bin_dir / "python3",
        """#!/bin/bash
if [ "$1" = "-c" ]; then
  exit 1
fi
exit 99
""",
    )
    _write_executable(
        bin_dir / "python3.11",
        f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
  echo "pip 24.0"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
  echo "python3.11" > "{chosen}"
  exit 0
fi
if [ "$1" = "-c" ]; then
  case "$2" in
    *"sys.version_info"*)
      exit 0
      ;;
    *'sys.prefix != getattr(sys, "base_prefix", sys.prefix)'*)
      echo 1
      exit 0
      ;;
    *'sysconfig.get_path("scripts")'*)
      echo "{bin_dir}"
      exit 0
      ;;
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
    *'resources.files("perfxpert")'*)
      echo "{bundled}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    _symlink_tool(bin_dir, "curl")
    _symlink_tool(bin_dir, "unzip")
    _symlink_tool(bin_dir, "git")
    _symlink_tool(bin_dir, "cat")
    _write_fake_perfxpert_code(bin_dir)

    env = _env_with_path(bin_dir)
    env["HOME"] = str(home_dir)
    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert chosen.read_text(encoding="utf-8").strip() == "python3.11"
