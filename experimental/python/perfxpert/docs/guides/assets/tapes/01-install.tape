# perfxpert getting-started demo — scenario 01-install
# Generated via: vhs 01-install.tape
# See docs/guides/getting-started.md
#
# The real `pip install` and `bash scripts/pip-install-from-git.sh`
# can't run inside the VHS sandbox (no network, no scripts/ cwd),
# so we install shell functions that shadow `pip` and `bash` and
# print canned output matching what a real install produces.
#
# The preamble writes a helper file (_install_funcs.sh) that
# defines the functions, sources it, then the user Types the real
# canonical commands — which end up calling the shadows.

Output ../gifs/01-install.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 650
Set Theme "TokyoNight"
Set TypingSpeed 25ms
Set Padding 20

Hide
Type "mkdir -p /tmp/perfxpert-vhs-demo && cd /tmp/perfxpert-vhs-demo"
Enter
Sleep 200ms
Type `cat > _install_funcs.sh <<'FUNCS_EOF'`
Enter
Type "pip() {"
Enter
Type `  if [ "$1" = "install" ]; then`
Enter
Type `    echo "Defaulting to user installation because normal site-packages is not writeable"`
Enter
Type `    echo "Requirement already satisfied: pip in ~/.local/lib/python3.10/site-packages (24.2)"`
Enter
Type `    echo "Requirement already satisfied: setuptools in ~/.local/lib/python3.10/site-packages (75.1.0)"`
Enter
Type `    echo "Requirement already satisfied: wheel in ~/.local/lib/python3.10/site-packages (0.44.0)"`
Enter
Type `    echo "Successfully installed pip-24.2 setuptools-75.1.0 wheel-0.44.0"`
Enter
Type "  else"
Enter
Type `    command pip "$@"`
Enter
Type "  fi"
Enter
Type "}"
Enter
Type "fakebash() {"
Enter
Type "  sleep 0.4"
Enter
Type `  echo "Cloning rocm-systems (shallow, no submodules) ... done in 14.8s"`
Enter
Type "  sleep 0.3"
Enter
Type `  echo "Initializing submodule opencode (scoped) ... done in 0.03s"`
Enter
Type "  sleep 0.2"
Enter
Type `  echo "Processing ./experimental/python/perfxpert"`
Enter
Type `  echo "  Preparing metadata (setup.py) ... done"`
Enter
Type "  sleep 0.3"
Enter
Type `  echo "Building wheels for collected packages: perfxpert"`
Enter
Type `  echo "  Building wheel for perfxpert (setup.py) ... done"`
Enter
Type "  sleep 0.2"
Enter
Type `  echo "Installing collected packages: perfxpert"`
Enter
Type `  echo "Successfully installed perfxpert-0.2.0"`
Enter
Type "}"
Enter
Type "bash() {"
Enter
Type `  if [ "$1" = "scripts/pip-install-from-git.sh" ]; then`
Enter
Type "    fakebash"
Enter
Type "  else"
Enter
Type `    command bash "$@"`
Enter
Type "  fi"
Enter
Type "}"
Enter
Type "FUNCS_EOF"
Enter
Type "source _install_funcs.sh"
Enter
Sleep 200ms
Type "clear"
Enter
Sleep 200ms
Show

Type "# Fast install (scoped submodule init — 30 ms vs 3 min)"
Enter
Sleep 600ms
Type "pip install -U pip setuptools wheel"
Enter
Sleep 1500ms
Type "bash scripts/pip-install-from-git.sh 4bdb610161"
Enter
Sleep 4s
