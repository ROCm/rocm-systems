# perfxpert getting-started demo — scenario 01-install
# Generated via: vhs 01-install.tape
# See docs/guides/getting-started.md
# Runs in a clean ROCm container via ../scripts/vhs-install-shell.sh.

Output ../gifs/01-install.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 650
Set Theme "TokyoNight"
Set TypingSpeed 25ms
Set Padding 20

Hide
Type "docker run --rm -it -v /home/aelwazir/work/ai-analysis-rocpd/rocm-systems-dev/.worktrees/perfxpert-phase10-guide-gif-refresh:/src:ro rocm/dev-ubuntu-22.04:7.2.2 bash -lc 'set -euo pipefail; export DEBIAN_FRONTEND=noninteractive; apt-get update >/dev/null; apt-get install -y --no-install-recommends git jq less ca-certificates >/dev/null; cp -a /src /tmp/src; exec bash --noprofile --norc'"
Enter
Sleep 25s
Type "cd /tmp/src"
Enter
Sleep 200ms
Type "clear"
Enter
Sleep 200ms
Show

Type "# Editable install in a clean ROCm dev container"
Enter
Sleep 600ms
Type "python3 -m pip install -U pip setuptools wheel"
Enter
Sleep 4s
Type "python3 -m pip install -e 'experimental/python/perfxpert[all]'"
Enter
Sleep 35s
Type "perfxpert --version"
Enter
Sleep 2s
Type "perfxpert-code opencode --help | head -5"
Enter
Sleep 4s
