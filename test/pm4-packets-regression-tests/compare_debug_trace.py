import re
from typing import Dict, List
import sys
import os

def parse_debug_trace_(filepath: str) -> Dict[str, List[str]]:
    """
    Parses a debug_trace.txt file and returns a dictionary mapping
    GPU block names to their list of records.
    """
    pmc_block_re = re.compile(r"^-+PMC DEBUG_TRACE for (.+?)\-+")
    pmc_block_re = re.compile(r"^-+SPM DEBUG_TRACE for (.+?)\-+")
    pmc_block_re = re.compile(r"^-+SQTT DEBUG_TRACE\-+")
    blocks = {}
    current_block = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            m = pmc_block_re.match(line)
            if m:
                current_block = m.group(1).replace("PMC_", "").replace("_dump.txt", "")
                blocks[current_block] = []
            elif current_block and not line.startswith("-") and line.strip():
                blocks[current_block].append(line)
    return blocks

def parse_debug_trace(filepath: str) -> Dict[str, List[str]]:
    """
    Parses a debug_trace.txt file and returns a dictionary mapping
    GPU block names to their list of records.
    Handles PMC, SPM, and SQTT patterns.
    """
    # Matches:
    #   PMC DEBUG_TRACE for PMC_SQ
    #   SPM DEBUG_TRACE for SPM_SQ
    #   SQTT DEBUG_TRACE
    block_re = re.compile(r"^-+(PMC|SPM) DEBUG_TRACE for (.+?)\-+|^-+(SQTT) DEBUG_TRACE\-+")
    blocks = {}
    current_block = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            m = block_re.match(line)
            if m:
                if m.group(1) in ("PMC", "SPM"):
                    current_block = m.group(2).replace("PMC_", "").replace("SPM_", "").replace("_dump.txt", "")
                elif m.group(3) == "SQTT":
                    current_block = "SQTT"
                blocks[current_block] = []
            elif current_block and not line.startswith("-") and line.strip():
                blocks[current_block].append(line)
    return blocks

def compare_blocks(blocks1: Dict[str, List[str]], blocks2: Dict[str, List[str]]) -> List[str]:
    """
    Compares two parsed debug_trace blocks and returns a list of difference messages.
    """
    messages = []
    all_blocks = set(blocks1.keys()) | set(blocks2.keys())
    for block in sorted(all_blocks):
        recs1 = blocks1.get(block, [])
        recs2 = blocks2.get(block, [])
        if recs1 != recs2:
            messages.append(f"Difference in GPU block: {block}")
            max_len = max(len(recs1), len(recs2))
            for i in range(max_len):
                line1 = recs1[i] if i < len(recs1) else "<missing>"
                line2 = recs2[i] if i < len(recs2) else "<missing>"
                if line1 != line2:
                    messages.append(f"  Command Packet #{i+1}:")
                    messages.append(f"    File1: {line1}")
                    messages.append(f"    File2: {line2}")
        else:
            messages.append(f"GPU block {block}: OK")
    return messages

def write_html_diff(block, recs1, recs2, html_file):
    import difflib
    differ = difflib.HtmlDiff()
    html = differ.make_file(recs1, recs2, fromdesc="File1", todesc="File2", context=False)
    with open(html_file, "a") as f:
        f.write(f"<h2>GPU block: {block}</h2>\n")
        f.write(html)

if __name__ == "__main__":
    usage = (
        "Usage: python compare_debug_trace_v2.py <file1> <file2> [--html [html_report.html]] [--rich]\n"
        "  --html [filename] : Generate HTML diff report (default: debug_trace_diff.html)\n"
        "  --rich            : Show colored diff in terminal (requires 'rich')\n"
        "  (default: plain text output)"
    )
    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)
    file1, file2 = sys.argv[1], sys.argv[2]
    mode = None
    html_file = "debug_trace_diff.html"
    if len(sys.argv) >= 4:
        if sys.argv[3] == "--html":
            mode = "--html"
            if len(sys.argv) == 5:
                html_file = sys.argv[4]
        elif sys.argv[3] == "--rich":
            mode = "--rich"
        else:
            print(usage)
            sys.exit(1)
    blocks1 = parse_debug_trace(file1)
    blocks2 = parse_debug_trace(file2)
    results = compare_blocks(blocks1, blocks2)
    if mode == "--html":
        report_title = os.path.splitext(os.path.basename(html_file))[0]
        report_title = report_title.replace("_", " ").replace("-", " ").upper()
        with open(html_file, "w") as f:
            f.write("<html><body>\n")
            f.write(f"<h1>{report_title}</h1>\n")
            f.write(f"<h3>File1(left): {file1}</h3>\n")
            f.write(f"<h3>File2(right): {file2}</h3>\n")
        exit_code = 0
        for block in sorted(set(blocks1.keys()) | set(blocks2.keys())):
            recs1 = blocks1.get(block, [])
            recs2 = blocks2.get(block, [])
            if recs1 != recs2:
                print("TEST FAILED : block", block, "has differences")
                write_html_diff(block, recs1, recs2, html_file)
                exit_code = 1
        with open(html_file, "a") as f:
            f.write("</body></html>\n")
        print(f"HTML diff report written to {html_file}")
        exit(exit_code)
    elif mode == "--rich":
        for msg in results:
            print(msg)
        for block in sorted(set(blocks1.keys()) | set(blocks2.keys())):
            recs1 = blocks1.get(block, [])
            recs2 = blocks2.get(block, [])
            if recs1 != recs2:
                print_rich_diff(block, recs1, recs2)
    else:
        for msg in results:
            print(msg)