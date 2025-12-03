import os
import re
import subprocess

base_ref = os.environ["BASE_REF"]

test_regex = r'TEST_CASE\("([^"]+)"(, "([^"]*)")*\)'
template_test_regex = r'TEMPLATE_TEST_CASE\("([^"]+)", "([^"]*)",\s*((?:[^;]*?))\)'
summary_path = "summary.txt"

test_pattern = re.compile(test_regex)
template_test_pattern = re.compile(template_test_regex, re.DOTALL)

diff = subprocess.check_output(
    ["git", "diff", f"origin/{base_ref}...HEAD", "--unified=0"],
    text=True,
    errors="replace",
)

problems = []

current_file = None
new_line = None

chunk_lines = []
chunk_start_line = None


def process_chunk():
    global problems, current_file, chunk_lines, chunk_start_line

    if not chunk_lines or chunk_start_line is None:
        return

    for i, content in enumerate(chunk_lines):
        line_no = chunk_start_line + i

        m = test_pattern.search(content)
        if m:
            try:
                g3 = m.group(3)
                if not g3 or not g3.strip():
                    raise IndexError
            except IndexError:
                problems.append((current_file, line_no, content.strip()))

    chunk_text = "\n".join(chunk_lines)

    for m in template_test_pattern.finditer(chunk_text):
        try:
            g2 = m.group(2)
            if not g2 or not g2.strip():
                raise IndexError
        except IndexError:
            prefix = chunk_text[: m.start()]
            line_offset = prefix.count("\n")
            line_no = chunk_start_line + line_offset

            match_lines = chunk_text[m.start() : m.end()].splitlines()
            snippet = match_lines[0].strip() if match_lines else ""

            problems.append((current_file, line_no, snippet))

    chunk_lines = []
    chunk_start_line = None


for line in diff.splitlines():
    if line.startswith("+++ b/"):
        process_chunk()
        current_file = line[6:]
    elif line.startswith("@@"):
        process_chunk()
        m = re.search(r"\+(\d+)(?:,(\d+))?", line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2) or "1")
        new_line = start - 1
    elif line.startswith("+") and not line.startswith("+++"):
        if new_line is None:
            continue
        new_line += 1
        content = line[1:]

        if not chunk_lines:
            chunk_start_line = new_line

        chunk_lines.append(content)
    else:
        process_chunk()

process_chunk()

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("### Test categories validation\n\n")
    if not problems:
        f.write("No issues found.\n")
    else:
        f.write("### Tests missing at least one category\n\n")
        f.write("| File | Line | Snippet |\n")
        f.write("| --- | --- | --- |\n")
        for file, line_no, snippet in problems:
            snippet = snippet.replace("|", "\\|")
            f.write(f"| `{file}` | {line_no} | `{snippet}` |\n")

raise SystemExit(bool(problems))
