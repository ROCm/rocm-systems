# perfxpert getting-started demo — scenario 02-help
# Generated via: vhs 02-help.tape
# See docs/guides/getting-started.md

Output ../gifs/02-help.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 700
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

Type "perfxpert --help"
Enter
Sleep 3s
Type "clear"
Enter
Sleep 300ms
Type "perfxpert analyze --help 2>&1 | head -30"
Enter
Sleep 4s
Type "clear"
Enter
Sleep 300ms
Type "perfxpert doctor 2>&1 | head -15"
Enter
Sleep 4s
