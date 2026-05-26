#!/usr/bin/env python3

import argparse
import collections
import json
import sys


def parse_expected(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def path_values(root, path):
    values = [root]
    for part in path.split("."):
        next_values = []
        for value in values:
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if not isinstance(candidate, dict) or part not in candidate:
                    continue
                child = candidate[part]
                if isinstance(child, list):
                    next_values.extend(child)
                else:
                    next_values.append(child)
        values = next_values
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Check required events and field values in a rocfuzz patch-report JSONL file."
    )
    parser.add_argument("report")
    parser.add_argument("--require-event", action="append", default=[])
    parser.add_argument("--require-count", nargs=2, action="append", default=[],
                        metavar=("EVENT", "COUNT"))
    parser.add_argument("--require-field", nargs=3, action="append", default=[],
                        metavar=("EVENT", "FIELD", "VALUE"))
    parser.add_argument("--require-min-field", nargs=3, action="append", default=[],
                        metavar=("EVENT", "FIELD", "MIN_VALUE"))
    parser.add_argument("--require-sum-field", nargs=3, action="append", default=[],
                        metavar=("EVENT", "FIELD", "MIN_SUM"))
    parser.add_argument("--require-nonempty-list", nargs=2, action="append", default=[],
                        metavar=("EVENT", "FIELD"))
    parser.add_argument("--require-list-field", nargs=4, action="append", default=[],
                        metavar=("EVENT", "LIST_FIELD", "ITEM_FIELD", "VALUE"))
    parser.add_argument("--require-list-item", nargs="+", action="append", default=[],
                        metavar="EVENT LIST_FIELD FIELD=VALUE")
    parser.add_argument("--require-list-item-count", nargs="+", action="append", default=[],
                        metavar="EVENT LIST_FIELD MIN_COUNT FIELD=VALUE")
    parser.add_argument("--require-list-item-delta", nargs="+", action="append", default=[],
                        metavar="EVENT LIST_FIELD NEW_FIELD OLD_FIELD MIN_DELTA FIELD=VALUE")
    parser.add_argument("--require-nonzero-hex", nargs=2, action="append", default=[],
                        metavar=("EVENT", "FIELD"))
    args = parser.parse_args()

    rows = []
    with open(args.report, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"{args.report}:{line_no}: invalid JSON: {exc}", file=sys.stderr)
                return 2

    counts = collections.Counter(row.get("event") for row in rows)
    print(dict(counts))

    missing = [event for event in args.require_event if counts[event] == 0]
    if missing:
        print(f"missing events: {', '.join(missing)}", file=sys.stderr)
        return 1

    for event, expected_text in args.require_count:
        try:
            expected = int(expected_text, 0)
        except ValueError:
            print(
                f"invalid count for event={event}: {expected_text!r}",
                file=sys.stderr,
            )
            return 2
        actual = counts[event]
        if actual != expected:
            print(
                f"wrong event count: event={event} expected={expected} actual={actual}",
                file=sys.stderr,
            )
            return 1

    for event, field, expected_text in args.require_field:
        expected = parse_expected(expected_text)
        matches = [
            row for row in rows
            if row.get("event") == event and row.get(field) == expected
        ]
        if not matches:
            print(
                f"missing field match: event={event} {field}={expected!r}",
                file=sys.stderr,
            )
            return 1

    for event, field, minimum_text in args.require_min_field:
        try:
            minimum = int(minimum_text, 0)
        except ValueError:
            print(
                f"invalid minimum for event={event} {field}: {minimum_text!r}",
                file=sys.stderr,
            )
            return 2
        matches = []
        for row in rows:
            if row.get("event") != event:
                continue
            value = row.get(field)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value >= minimum:
                matches.append(row)
        if not matches:
            print(
                f"missing minimum field: event={event} {field}>={minimum}",
                file=sys.stderr,
            )
            return 1

    for event, field, minimum_text in args.require_sum_field:
        try:
            minimum = int(minimum_text, 0)
        except ValueError:
            print(
                f"invalid minimum sum for event={event} {field}: {minimum_text!r}",
                file=sys.stderr,
            )
            return 2
        total = 0
        for row in rows:
            if row.get("event") != event:
                continue
            value = row.get(field)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                total += value
        if total < minimum:
            print(
                f"field sum too small: event={event} {field} sum={total} minimum={minimum}",
                file=sys.stderr,
            )
            return 1

    for event, field in args.require_nonempty_list:
        matches = [
            row for row in rows
            if row.get("event") == event
            and len(path_values(row, field)) > 0
        ]
        if not matches:
            print(
                f"missing nonempty list: event={event} {field}",
                file=sys.stderr,
            )
            return 1

    for event, list_field, item_field, expected_text in args.require_list_field:
        expected = parse_expected(expected_text)
        matches = []
        for row in rows:
            if row.get("event") != event:
                continue
            for item in path_values(row, list_field):
                if isinstance(item, dict) and item.get(item_field) == expected:
                    matches.append(row)
                    break
        if not matches:
            print(
                "missing list field match: "
                f"event={event} {list_field}[].{item_field}={expected!r}",
                file=sys.stderr,
            )
            return 1

    for values in args.require_list_item:
        if len(values) < 3:
            print(
                "invalid --require-list-item: expected EVENT LIST_FIELD FIELD=VALUE...",
                file=sys.stderr,
            )
            return 2
        event = values[0]
        list_field = values[1]
        requirements = {}
        for item in values[2:]:
            if "=" not in item:
                print(
                    f"invalid --require-list-item requirement: {item!r}",
                    file=sys.stderr,
                )
                return 2
            field, expected_text = item.split("=", 1)
            if not field:
                print(
                    f"invalid --require-list-item empty field: {item!r}",
                    file=sys.stderr,
                )
                return 2
            requirements[field] = parse_expected(expected_text)
        matches = []
        for row in rows:
            if row.get("event") != event:
                continue
            for item in path_values(row, list_field):
                if not isinstance(item, dict):
                    continue
                if all(item.get(field) == expected for field, expected in requirements.items()):
                    matches.append(row)
                    break
        if not matches:
            required = " ".join(
                f"{field}={expected!r}" for field, expected in sorted(requirements.items())
            )
            print(
                "missing list item match: "
                f"event={event} {list_field}[] with {required}",
                file=sys.stderr,
            )
            return 1

    for values in args.require_list_item_count:
        if len(values) < 4:
            print(
                "invalid --require-list-item-count: "
                "expected EVENT LIST_FIELD MIN_COUNT FIELD=VALUE...",
                file=sys.stderr,
            )
            return 2
        event = values[0]
        list_field = values[1]
        try:
            minimum = int(values[2], 0)
        except ValueError:
            print(
                f"invalid --require-list-item-count minimum: {values[2]!r}",
                file=sys.stderr,
            )
            return 2
        requirements = {}
        for item in values[3:]:
            if "=" not in item:
                print(
                    f"invalid --require-list-item-count requirement: {item!r}",
                    file=sys.stderr,
                )
                return 2
            field, expected_text = item.split("=", 1)
            if not field:
                print(
                    f"invalid --require-list-item-count empty field: {item!r}",
                    file=sys.stderr,
                )
                return 2
            requirements[field] = parse_expected(expected_text)
        count = 0
        for row in rows:
            if row.get("event") != event:
                continue
            for item in path_values(row, list_field):
                if not isinstance(item, dict):
                    continue
                if all(item.get(field) == expected
                       for field, expected in requirements.items()):
                    count += 1
        if count < minimum:
            required = " ".join(
                f"{field}={expected!r}" for field, expected in sorted(requirements.items())
            )
            print(
                "list item count too small: "
                f"event={event} {list_field}[] count={count} minimum={minimum} "
                f"with {required}",
                file=sys.stderr,
            )
            return 1

    for values in args.require_list_item_delta:
        if len(values) < 5:
            print(
                "invalid --require-list-item-delta: "
                "expected EVENT LIST_FIELD NEW_FIELD OLD_FIELD MIN_DELTA FIELD=VALUE...",
                file=sys.stderr,
            )
            return 2
        event = values[0]
        list_field = values[1]
        new_field = values[2]
        old_field = values[3]
        try:
            minimum = int(values[4], 0)
        except ValueError:
            print(
                f"invalid --require-list-item-delta minimum: {values[4]!r}",
                file=sys.stderr,
            )
            return 2
        requirements = {}
        for item in values[5:]:
            if "=" not in item:
                print(
                    f"invalid --require-list-item-delta requirement: {item!r}",
                    file=sys.stderr,
                )
                return 2
            field, expected_text = item.split("=", 1)
            if not field:
                print(
                    f"invalid --require-list-item-delta empty field: {item!r}",
                    file=sys.stderr,
                )
                return 2
            requirements[field] = parse_expected(expected_text)
        matches = []
        for row in rows:
            if row.get("event") != event:
                continue
            for item in path_values(row, list_field):
                if not isinstance(item, dict):
                    continue
                if not all(item.get(field) == expected
                           for field, expected in requirements.items()):
                    continue
                old_value = item.get(old_field)
                new_value = item.get(new_field)
                if isinstance(old_value, bool) or isinstance(new_value, bool):
                    continue
                if not isinstance(old_value, int) or not isinstance(new_value, int):
                    continue
                if new_value - old_value >= minimum:
                    matches.append(row)
                    break
        if not matches:
            required = " ".join(
                f"{field}={expected!r}" for field, expected in sorted(requirements.items())
            )
            suffix = f" with {required}" if required else ""
            print(
                "missing list item delta: "
                f"event={event} {list_field}[] {new_field}-{old_field}>={minimum}"
                f"{suffix}",
                file=sys.stderr,
            )
            return 1

    for event, field in args.require_nonzero_hex:
        matches = []
        for row in rows:
            if row.get("event") != event:
                continue
            value = row.get(field)
            if isinstance(value, str):
                try:
                    if int(value, 0) != 0:
                        matches.append(row)
                except ValueError:
                    pass
            elif isinstance(value, int) and value != 0:
                matches.append(row)
        if not matches:
            print(
                f"missing nonzero hex field: event={event} {field}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
