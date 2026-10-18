# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import test_api_contract as contract

DIRECTIVE = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*(?:\n|$)", re.M)
FUNCTION = re.compile(r"([^(){};]+?)\b(amdsmi_\w+)\s*\(([^()]*)\)\s*;", re.S)
CTYPE = r"[A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)*(?:\s*\*\s*(?:(?:const|volatile|restrict)\b\s*)?)*"
PARAMETER = re.compile(r"(" + CTYPE + r")(?:\s+|(?<=\*))\w+\s*((?:\[\s*\w*\s*\]\s*)*)")
TYPE = re.compile(CTYPE)
PUNCTUATION = re.compile(r"\s*([*\[\],;{}()=])\s*")
GO_FUNCTION = re.compile(r"\bfunc\s+(\w+)\s*(?:\[[^\]]*\])?\s*\([^{}]*?\{")
C_HELPER = re.compile(r"\b(go_amdsmi_\w+)\s*\([^;{}]*\)\s*\{")
CALL = re.compile(r"(?<![\w.])(?:C\s*\.\s*)?(\w+)\s*(?:\[[^\]\n]*\])?\s*\(")
IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
TYPEDEF_NAME = re.compile(r"\b([A-Za-z_]\w*)\s*;$")
MACRO = re.compile(r"^[ \t]*#define[ \t]+([A-Za-z_]\w*)\b(.*)$", re.M)


def normalize(text: str) -> str:
    return PUNCTUATION.sub(r"\1", " ".join(text.split()))


def header_statements(header: str, *, positions: bool = False) -> list:
    header = contract.SOURCE_TOKENS.sub(
        lambda m: " " * len(m.group()) if m.group().startswith(("/*", "//")) else m.group(), header
    )
    header = DIRECTIVE.sub(lambda m: " " * len(m.group()), header)
    header, linkage = re.subn(r'extern\s+"C"\s*\{', lambda m: " " * len(m.group()), header)
    if linkage:
        header, closing = re.subn(r"\}\s*$", lambda m: " " * len(m.group()), header)
        if linkage != 1 or closing != 1:
            raise ValueError("unparsed C linkage")
    # Tokenize literals atomically; braces and semicolons inside strings are not syntax.
    tokens = re.finditer(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{};]', header)
    depth, start, statements = 0, 0, []
    for token in tokens:
        text = token.group()
        if text == "{":
            depth += 1
        elif text == "}":
            depth -= 1
            if depth < 0:
                raise ValueError("unparsed closing brace")
        elif text == ";" and depth == 0:
            statement = header[start : token.end()].strip()
            begin = (
                start + len(header[start : token.end()]) - len(header[start : token.end()].lstrip())
            )
            statements.append((statement, begin, token.end()) if positions else statement)
            start = token.end()
    if depth or header[start:].strip():
        raise ValueError("unparsed trailing declaration: " + header[start:].strip()[:100])
    return statements


def signature(statement: str) -> tuple:
    match = FUNCTION.fullmatch(statement)
    if not match or not TYPE.fullmatch(match[1].strip()):
        raise ValueError("unparsed public declaration: " + statement[:150])
    parameters = []
    arguments = match[3].strip()
    for argument in [] if arguments == "void" else arguments.split(","):
        parameter = PARAMETER.fullmatch(argument.strip())
        if not parameter:
            raise ValueError("unparsed parameter in " + match[2] + ": " + argument)
        parameters.append(normalize(parameter[1] + parameter[2]))
    return match[2], normalize(match[1]) + "(" + ",".join(parameters) + ")"


def declarations(header: str) -> dict:
    result = {}
    for statement in header_statements(header):
        if statement.startswith("typedef "):
            continue
        if re.fullmatch(
            r"static\s+char\s*\*\s*const\s+amdsmi_hsmp_freqlimit_src_names\[\]\s*=\s*\{[^{}]*\}\s*;",
            statement,
            re.S,
        ):
            continue
        name, value = signature(statement)
        if name in result:
            raise ValueError("duplicate public declaration: " + name)
        result[name] = value
    return result


def function_calls(source: str, pattern: re.Pattern) -> dict:
    functions = {}
    for match in pattern.finditer(source):
        depth = 1
        end = match.end()
        while depth and end < len(source):
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        if depth or match[1] in functions:
            raise ValueError("unparsed or duplicate Go/helper mapping: " + match[1])
        functions[match[1]] = set(CALL.findall(source[match.end() : end - 1]))
    return functions


def reachable(name: str, graph: dict) -> set:
    found, pending = set(), [name]
    while pending:
        node = pending.pop()
        if node not in found:
            found.add(node)
            pending.extend(graph.get(node, ()))
    return found


def unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate inventory key: " + key)
        result[key] = value
    return result


def type_fingerprints(header: str, source: str) -> dict:
    definitions = {}
    spans = {}
    for statement, start, end in header_statements(header, positions=True):
        if statement.startswith("typedef "):
            match = TYPEDEF_NAME.search(statement)
            if not match or match[1] in definitions:
                raise ValueError("unparsed or duplicate typedef: " + statement[:100])
            definitions[match[1]] = statement
            spans[match[1]] = (start, end)
    clean = contract.COMMENTS.sub(" ", header).replace("\\\n", " ")
    definitions.update({name: body.strip() for name, body in MACRO.findall(clean)})
    roots = set(re.findall(r"\bC\s*\.\s*(\w+)", contract.source_code(source)))
    for token in contract.SOURCE_TOKENS.finditer(source):
        if token.group().startswith("/*") and contract.C_IMPORT.match(source, token.end()):
            roots.update(IDENTIFIER.findall(contract.source_code(token.group()[2:-2])))
    functions = declarations(header)
    for name in contract.native_calls(source):
        roots.update(IDENTIFIER.findall(functions.get(name, "")))
    for name, body in definitions.items():
        if body.startswith("typedef enum") and roots & set(IDENTIFIER.findall(body)):
            roots.add(name)
    graph = {
        name: set(IDENTIFIER.findall(body)) & definitions.keys()
        for name, body in definitions.items()
    }
    used = set()
    for name in roots & definitions.keys():
        used.update(reachable(name, graph))
    fingerprints = {
        name: hashlib.sha256(normalize(definitions[name]).encode()).hexdigest()
        for name in sorted(used)
    }
    # Anchor directives to bound types, not unrelated declarations elsewhere.
    masked = contract.COMMENTS.sub(lambda m: " " * len(m.group()), header)
    directives = list(DIRECTIVE.finditer(masked))
    context = []
    for match in directives:
        if not re.match(
            r"\s*#\s*(?:if|ifdef|ifndef|elif|else|endif|pragma|include|undef)\b", match.group()
        ):
            continue
        before = sorted(name for name in used & spans.keys() if spans[name][1] <= match.start())
        inside = [
            (
                name,
                len(re.findall(r"\w+|[^\w\s]", DIRECTIVE.sub("", masked[start : match.start()]))),
            )
            for name, (start, end) in sorted(spans.items())
            if name in used and start <= match.start() < end
        ]
        context.append((normalize(match.group()), before, inside))
    if context:
        fingerprints["_preprocessor_context"] = hashlib.sha256(
            json.dumps(context).encode()
        ).hexdigest()
    return fingerprints


def check_inventory(manifest: dict) -> None:
    if not isinstance(manifest, dict) or not all(
        isinstance(manifest.get(key), dict) for key in ("apis", "reasons", "types")
    ):
        raise ValueError("invalid inventory sections")
    for key, reason in manifest["reasons"].items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("empty inventory reason: " + key)
    for name, row in manifest["apis"].items():
        if not isinstance(row, dict) or not all(isinstance(v, str) for v in row.values()):
            raise ValueError("invalid inventory row: " + name)
        required = {"signature", "status", "reason"}
        allowed = required | ({"go", "via"} if row.get("status") == "bound" else set())
        if (
            not required <= row.keys() <= allowed
            or row["status"] not in {"bound", "deferred"}
            or row["reason"] not in manifest["reasons"]
            or not row["signature"].strip()
        ):
            raise ValueError("invalid inventory status/reason/fields: " + name)
        if row["status"] == "bound" and not row.get("go", "").strip():
            raise ValueError("missing inventory Go mapping: " + name)


def check_mappings(source: str, apis: dict) -> None:
    graph = function_calls(contract.source_code(source), GO_FUNCTION)
    exports = {name for name in graph if name[0].isupper()}
    for token in contract.SOURCE_TOKENS.finditer(source):
        if token.group().startswith("/*") and contract.C_IMPORT.match(source, token.end()):
            graph.update(function_calls(contract.source_code(token.group()[2:-2]), C_HELPER))
    bound = {name for name, row in apis.items() if row["status"] == "bound"}
    calls = contract.native_calls(source)
    if calls != bound:
        raise ValueError(
            "native calls differ from bound inventory: " + ", ".join(sorted(calls ^ bound))
        )
    for name in bound:
        row = apis[name]
        export, via = row.get("go", ""), row.get("via", row.get("go", ""))
        if (
            export not in exports
            or via not in graph
            or via not in reachable(export, graph)
            or name not in reachable(via, graph)
        ):
            raise ValueError("stale Go mapping: " + name + " -> " + export)


def check_coverage(*, header: str, source: str, manifest: dict) -> None:
    check_inventory(manifest)
    current = declarations(header)
    new = current.keys() - manifest["apis"].keys()
    if new:
        raise ValueError("new declarations: " + ", ".join(sorted(new)))
    removed = manifest["apis"].keys() - current.keys()
    if removed:
        raise ValueError("removed declarations: " + ", ".join(sorted(removed)))
    for name, signature in current.items():
        if signature != manifest["apis"][name]["signature"]:
            raise ValueError("changed declaration: " + name + " -> " + signature)
    check_mappings(source, manifest["apis"])
    types = type_fingerprints(header, source)
    if types != manifest["types"]:
        new = types.keys() - manifest["types"].keys()
        removed = manifest["types"].keys() - types.keys()
        changed = {
            name
            for name in types.keys() & manifest["types"].keys()
            if types[name] != manifest["types"][name]
        }
        raise ValueError(
            "binding types changed: new="
            + str(sorted(new))
            + "; removed="
            + str(sorted(removed))
            + "; changed="
            + str(sorted(changed))
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the complete public C inventory with Go binding presence"
    )
    parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    try:
        manifest = json.loads(
            Path(__file__).with_name("api_coverage.json").read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
        header = (project / "include/amd_smi/amdsmi.h").read_text(encoding="utf-8")
        sources = {
            path: path.read_text(encoding="utf-8")
            for path in sorted((project / "go").rglob("*.go"))
            if not (
                path.name.endswith("_test.go")
                or path.name == "mock_bridge_linux.go"
                or "testdata" in path.relative_to(project / "go").parts
            )
        }
        calls = set().union(*(contract.native_calls(text) for text in sources.values()))
        contract.check_native_calls(calls, available_only=False)
        source = "\n".join(
            text for path, text in sources.items() if path.parent == project / "go/amdsmi"
        )
        check_coverage(header=header, source=source, manifest=manifest)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    counts = Counter(row["status"] for row in manifest["apis"].values())
    print(
        "Go coverage inventory passed: public declarations: {}; bound: {}; deferred: {}".format(
            len(manifest["apis"]), counts["bound"], counts["deferred"]
        )
    )
    print(
        "Binding type/macro fingerprints: {}; binding presence, not consumer readiness.".format(
            len(manifest["types"])
        )
    )
    for reason, count in sorted(
        Counter(row["reason"] for row in manifest["apis"].values()).items()
    ):
        print("  {}: {}: {}".format(reason, count, manifest["reasons"][reason]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
