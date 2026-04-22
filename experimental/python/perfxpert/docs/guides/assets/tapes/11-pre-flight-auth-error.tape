# perfxpert getting-started demo — scenario 11-pre-flight-auth-error
# Generated via: vhs 11-pre-flight-auth-error.tape
# See docs/guides/getting-started.md

Output ../gifs/11-pre-flight-auth-error.gif

Set Shell "./vhs-demo-shell.sh"
Set FontSize 14
Set Width 1200
Set Height 500
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

Type "# Pre-flight auth check: fails fast, no empty output file"
Enter
Sleep 500ms
Type "unset ANTHROPIC_API_KEY PERFXPERT_LLM_ANTHROPIC_KEY"
Enter
Sleep 300ms
Type "set +e"
Enter
Sleep 300ms
Type "perfxpert analyze -i trace.db --llm anthropic --format webview -d . -o preflight"
Enter
Sleep 4s
Type "echo rc=$?"
Enter
Sleep 800ms
Type "test ! -f preflight.html && echo 'no output file written'"
Enter
Sleep 2s
