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
_INSTALL_WRAPPER = _APP_ROOT / "scripts" / "pip-install-from-git.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _env_with_path(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(path)
    return env


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
    assert "python3 -m venv .venv" in result.stdout
    assert "python3-venv" in result.stdout
    assert "python3-pip" in result.stdout
    assert "git" in result.stdout
    assert "bun.sh/install" in result.stdout
    assert "It never downloads" in result.stdout
    assert "dnf install -y curl git python3.11 python3.11-pip" in result.stdout
    assert "dnf install -y curl git python3 python3-pip" in result.stdout
    assert "zypper install -y curl git python311 python311-pip" in result.stdout
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
    assert "bun.sh/install" in result.stdout
    assert "It never downloads" in result.stdout
    assert "dnf install -y curl git python3.11 python3.11-pip" in result.stdout
    assert "zypper install -y curl git python311 python311-pip" in result.stdout
    assert 'REF=<SHA>; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"' in result.stdout


def test_readme_keeps_customer_install_flow_curl_only() -> None:
    text = _README.read_text(encoding="utf-8")
    assert "curl" in text
    assert "curl -fsSL https://bun.sh/install | bash" in text
    assert "python3 -m venv .venv" in text
    assert "python3-venv" in text
    assert "python3-pip" in text
    assert "never downloads a separate Python runtime" in text
    assert 'REF=<SHA>; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"' in text
    assert "GIT_CONFIG_COUNT=1" not in text
    assert "git clone --depth 1 --no-recurse-submodules" not in text
    assert "wget -qO-" not in text


def test_getting_started_keeps_internal_install_detail() -> None:
    text = _GETTING_STARTED.read_text(encoding="utf-8")
    assert "curl" in text
    assert "curl -fsSL https://bun.sh/install | bash" in text
    assert "python3 -m venv .venv" in text
    assert "python3-venv" in text
    assert "python3-pip" in text
    assert "never downloads a separate Python runtime" in text
    assert "dnf install -y curl git python3.11 python3.11-pip" in text
    assert "dnf install -y curl git python3 python3-pip" in text
    assert "zypper install -y curl git python311 python311-pip" in text
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
    for doc in (_README, _GETTING_STARTED):
        text = doc.read_text(encoding="utf-8")
        assert "perfxpert-code install-patches" in text, (
            f"{doc} must explain how to enable perfxpert-code after a bun-less install."
        )
        assert "curl -fsSL https://bun.sh/install | bash" in text, (
            f"{doc} must include the bun install command."
        )
        assert "opencode.ai/install" not in text, (
            f"{doc} must not send users to an upstream opencode curl installer."
        )


def test_install_wrapper_fails_cleanly_when_git_missing(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    os.symlink(shutil.which("python3"), bin_dir / "python3")
    os.symlink(shutil.which("cat"), bin_dir / "cat")

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER)],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "`git` is required" in result.stderr
    assert "apt install -y curl git python3-venv python3-pip" in result.stderr
    assert "dnf install -y curl git python3.11 python3.11-pip" in result.stderr
    assert "dnf install -y curl git python3 python3-pip" in result.stderr
    assert "zypper install -y curl git python311 python311-pip" in result.stderr


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
    os.symlink(shutil.which("cat"), bin_dir / "cat")

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER)],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "Python 3.10+ is required" in result.stderr


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
    os.symlink(shutil.which("git"), bin_dir / "git")
    os.symlink(shutil.which("cat"), bin_dir / "cat")

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER)],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "`python -m pip` is unavailable" in result.stderr
    assert "apt install -y curl git python3-venv python3-pip" in result.stderr
    assert "dnf install -y curl git python3.11 python3.11-pip" in result.stderr
    assert "dnf install -y curl git python3 python3-pip" in result.stderr
    assert "zypper install -y curl git python311 python311-pip" in result.stderr


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
    *'sysconfig.get_path("stdlib")'*)
      echo "{managed}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    os.symlink(shutil.which("cat"), bin_dir / "cat")
    os.symlink(shutil.which("git"), bin_dir / "git")

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
    assert "dnf install -y curl git python3.11 python3.11-pip" in result.stderr
    assert "dnf install -y curl git python3 python3-pip" in result.stderr
    assert "zypper install -y curl git python311 python311-pip" in result.stderr


def test_install_wrapper_reports_bunless_tui_follow_up(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
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
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    os.symlink(shutil.which("git"), bin_dir / "git")
    os.symlink(shutil.which("cat"), bin_dir / "cat")

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 0
    assert "perfxpert-code install-patches" in result.stderr
    assert "bun.sh/install" in result.stderr


def test_install_wrapper_prefers_active_python_before_versioned_fallbacks(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    chosen = tmp_path / "chosen.txt"

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
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
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
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    os.symlink(shutil.which("git"), bin_dir / "git")
    os.symlink(shutil.which("cat"), bin_dir / "cat")

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert chosen.read_text(encoding="utf-8").strip() == "python"


def test_install_wrapper_falls_back_to_supported_versioned_python(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    chosen = tmp_path / "chosen.txt"

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
    *'sysconfig.get_path("stdlib")'*)
      echo "{tmp_path / 'missing-externally-managed'}"
      exit 0
      ;;
  esac
fi
exit 99
""",
    )
    os.symlink(shutil.which("git"), bin_dir / "git")
    os.symlink(shutil.which("cat"), bin_dir / "cat")

    result = subprocess.run(
        ["/bin/bash", str(_INSTALL_WRAPPER), "develop"],
        capture_output=True,
        text=True,
        env=_env_with_path(bin_dir),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert chosen.read_text(encoding="utf-8").strip() == "python3.11"
