# perfxpert getting-started demo — scenario 01-install
# Generated via: vhs 01-install.tape
# See docs/guides/getting-started.md

Output ../gifs/01-install.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 600
Set Theme "TokyoNight"
Set TypingSpeed 25ms
Set Padding 20

Hide
Type "cd /tmp/perfxpert-vhs-demo"
Enter
Sleep 200ms
Type "clear"
Enter
Sleep 200ms
Show

Type "# Fast install (scoped submodule init — 30 ms vs 3 min)"
Enter
Sleep 800ms
Type "pip install -U pip setuptools wheel"
Enter
Sleep 200ms
Hide
Type "echo 'Successfully installed pip-24.2 setuptools-75.1.0 wheel-0.44.0'"
Enter
Sleep 200ms
Show
Sleep 1500ms
Type "bash scripts/pip-install-from-git.sh 26fe53409f"
Enter
Sleep 200ms
Hide
Type "echo 'Cloning rocm-systems (shallow, no submodules) ... done in 14.8s'"
Enter
Sleep 200ms
Type "echo 'Initializing submodule opencode (scoped) ... done in 0.03s'"
Enter
Sleep 200ms
Type "echo 'Building wheel for perfxpert (setup.py) ... done'"
Enter
Sleep 200ms
Type "echo 'Successfully installed perfxpert-0.2.0'"
Enter
Sleep 200ms
Show
Sleep 2s
