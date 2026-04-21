# perfxpert getting-started demo — scenario 15-all-providers
# Generated via: vhs 15-all-providers.tape
# See docs/guides/getting-started.md

Output ../gifs/15-all-providers.gif

Set Shell "bash"
Set FontSize 13
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

Type "# 5 LLM providers, one flag: --llm {anthropic,openai,ollama,private,opencode}"
Enter
Sleep 500ms
Type "perfxpert analyze --help 2>&1 | grep -A 10 -- '--llm ' | head -12"
Enter
Sleep 7s
