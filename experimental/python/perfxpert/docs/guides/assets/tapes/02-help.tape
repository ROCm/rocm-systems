# perfxpert getting-started demo — scenario 02-help
# Generated via: vhs 02-help.tape
# See docs/guides/getting-started.md
#
# Shows `perfxpert --help` then `perfxpert analyze --help | head -30`.
# The trailing `doctor` demo was dropped (it's redundant with 03-doctor
# and was clearing the useful help text off-screen — bad for the
# last-frame preview users see when the GIF is paused).

Output ../gifs/02-help.gif

Set Shell "bash"
Set FontSize 13
Set Width 1200
Set Height 750
Set Theme "TokyoNight"
Set TypingSpeed 25ms
Set Padding 20

Hide
Type "cd /tmp/perfxpert-vhs-demo 2>/dev/null || cd /tmp"
Enter
Sleep 200ms
Type "clear"
Enter
Sleep 200ms
Show

Type "perfxpert --help"
Enter
Sleep 4s
Type "clear"
Enter
Sleep 300ms
Type "perfxpert analyze --help 2>&1 | head -30"
Enter
Sleep 7s
