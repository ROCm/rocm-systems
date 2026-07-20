#!/usr/bin/env python3
"""Generate the per-provider LTTng curated-category manifest (schema v1).

Reads the pattern-based category taxonomy (lttng_categories.yaml) and the
*generated* tp.h header for a provider (the ground truth for exactly which
event names exist, including chunk-suffixed events like `<api>_2`), and
writes a JSON manifest mapping each category name to the full list of
concrete, enable-able LTTng event names.

Why parse tp.h instead of curated_apis.yaml directly: schema v1 emits one
LTTng event per curated API, but a "big" API (more than the 10-field LTTng
budget) is split by lttng_curated_lib.event_field_chunks() into multiple
chunk events `<api>`, `<api>_2`, `<api>_3`, ... Enabling an API's tracing
means enabling *every* one of its chunk events. The generated tp.h is the
single source of truth for exactly which chunk events a given API produced,
so this tool parses it rather than re-deriving the chunk split.

Usage:
    python3 lttng_gen_category_manifest.py \\
        --provider hip \\
        --taxonomy shared/lttng/scripts/lttng_categories.yaml \\
        --tp-header projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h \\
        --out projects/clr/hipamd/scripts/lttng_categories_manifest.json

CI / pre-commit usage -- verify the checked-in manifest is still exactly
what the generator produces from the checked-in taxonomy + tp.h:

    python3 lttng_gen_category_manifest.py --provider hip --check \\
        --taxonomy ... --tp-header ... --out ...

`--check` generates in memory, diffs the result against the on-disk `--out`
file, prints a unified diff on mismatch, and exits 1. It never overwrites
the target file.
"""
import argparse
import difflib
import fnmatch
import json
import os
import re
import sys

import yaml

PROVIDER_TAG = {
    'hip': 'rocm_hip',
    'hsa': 'rocm_hsa',
}

# Matches `LTTNG_UST_TRACEPOINT_EVENT(\n    rocm_<prov>, <event_name>,` (the
# generator always emits the opening paren, provider tag, and event name on
# a fixed two-line pattern; tolerate arbitrary whitespace between tokens).
_EVENT_RE = re.compile(
    r'LTTNG_UST_TRACEPOINT_EVENT\s*\(\s*(\w+)\s*,\s*(\w+)\s*,')

# A chunk-suffixed event name ends in a literal underscore + one-or-more
# digits, e.g. `hipMemcpy3DBatchAsync_2`. Real API names never end this way
# (verified against both curated_apis.yaml files at generator-review time),
# so this is safe to use unconditionally rather than cross-checking against
# a known-API set.
_CHUNK_SUFFIX_RE = re.compile(r'^(.*)_(\d+)$')


def parse_tp_header(path, provider_tag):
    """Return the list of event names declared for `provider_tag` in the
    generated tp.h at `path`, in file order."""
    with open(path) as f:
        text = f.read()
    events = []
    for tag, name in _EVENT_RE.findall(text):
        if tag == provider_tag:
            events.append(name)
    if not events:
        sys.exit(f"ERROR: no '{provider_tag}' LTTNG_UST_TRACEPOINT_EVENT "
                  f"declarations found in {path}")
    return events


def base_api_of(event_name):
    """Strip a trailing chunk suffix (`_2`, `_3`, ...) from an event name to
    recover its base curated API name. Names with no such suffix are
    returned unchanged."""
    m = _CHUNK_SUFFIX_RE.match(event_name)
    return m.group(1) if m else event_name


def group_events_by_base_api(events):
    """Return {base_api: [event names incl. chunks, in file order]}."""
    by_base = {}
    for ev in events:
        by_base.setdefault(base_api_of(ev), []).append(ev)
    return by_base


class TaxonomyError(Exception):
    """Malformed taxonomy, empty category, or a pattern matching no APIs."""


def _validate_taxonomy_shape(provider, categories_cfg):
    """Structural validation of the provider's taxonomy section: it must be a
    mapping of category-name -> non-empty list of non-empty string patterns."""
    if not isinstance(categories_cfg, dict):
        raise TaxonomyError(
            f"taxonomy section for provider {provider!r} must be a mapping of "
            f"category -> [patterns], got {type(categories_cfg).__name__}")
    if not categories_cfg:
        raise TaxonomyError(
            f"taxonomy section for provider {provider!r} is empty (no "
            f"categories defined)")
    for cat_name, patterns in categories_cfg.items():
        if not isinstance(patterns, list) or not patterns:
            raise TaxonomyError(
                f"category {cat_name!r}: patterns must be a non-empty list, "
                f"got {patterns!r}")
        for pat in patterns:
            if not isinstance(pat, str) or not pat.strip():
                raise TaxonomyError(
                    f"category {cat_name!r}: each pattern must be a non-empty "
                    f"string, got {pat!r}")


def build_manifest(provider, taxonomy, events, allow_uncategorized=False):
    """Build the manifest dict for `provider` from the taxonomy (already
    loaded from YAML) and the flat list of event names parsed from tp.h.

    Enforces category-coverage / taxonomy-quality invariants (raising
    TaxonomyError) so a stale or sloppy taxonomy fails the gate rather than
    silently producing a weaker manifest:
      - the provider's taxonomy section must be well-formed (mapping of
        category -> non-empty list of non-empty string patterns);
      - no category may be empty (every category must match >=1 API);
      - no pattern may match zero APIs (a dead pattern is almost always a
        typo or a since-renamed API);
      - every curated API must fall into >=1 category, unless
        `allow_uncategorized` is set (explicit escape hatch)."""
    by_base = group_events_by_base_api(events)
    base_apis = sorted(by_base)
    categories_cfg = taxonomy.get(provider, {})
    _validate_taxonomy_shape(provider, categories_cfg)

    categories = {}
    categorized_bases = set()
    empty_categories = []
    dead_patterns = []
    for cat_name in sorted(categories_cfg):
        patterns = categories_cfg[cat_name]
        matched_bases = []
        for pat in patterns:
            pat_matches = [base for base in base_apis
                           if fnmatch.fnmatchcase(base, pat)]
            if not pat_matches:
                dead_patterns.append((cat_name, pat))
            matched_bases.extend(pat_matches)
        # De-dup while preserving base_apis (sorted) order.
        matched_bases = sorted(set(matched_bases))
        if not matched_bases:
            empty_categories.append(cat_name)
        categorized_bases.update(matched_bases)
        cat_events = []
        for base in matched_bases:
            cat_events.extend(by_base[base])
        categories[cat_name] = cat_events

    uncategorized_apis = sorted(set(base_apis) - categorized_bases)

    problems = []
    if empty_categories:
        problems.append("empty categories (match zero APIs): "
                        + ', '.join(empty_categories))
    if dead_patterns:
        problems.append("patterns matching zero APIs: "
                        + ', '.join(f"{c}:{p!r}" for c, p in dead_patterns))
    if uncategorized_apis and not allow_uncategorized:
        problems.append(
            f"{len(uncategorized_apis)} uncategorized API(s) (pass "
            f"--allow-uncategorized to permit): "
            + ', '.join(uncategorized_apis))
    if problems:
        raise TaxonomyError(
            f"provider {provider!r} taxonomy/coverage errors:\n  - "
            + "\n  - ".join(problems))

    manifest = {
        'provider': PROVIDER_TAG[provider],
        'categories': categories,
        'all_apis': list(events),
        'uncategorized_apis': uncategorized_apis,
    }
    return manifest, by_base


def render(manifest):
    return json.dumps(manifest, indent=2, sort_keys=True) + '\n'


def print_summary(provider, manifest, by_base, file=sys.stderr):
    print(f"=== {provider} category manifest summary ===", file=file)
    for cat_name in sorted(manifest['categories']):
        events = manifest['categories'][cat_name]
        bases = sorted({base_api_of(e) for e in events})
        print(f"  {cat_name}: {len(bases)} APIs, {len(events)} events",
              file=file)
    uncategorized = manifest['uncategorized_apis']
    print(f"  TOTAL base APIs: {len(by_base)}  "
          f"categorized: {len(by_base) - len(uncategorized)}  "
          f"uncategorized: {len(uncategorized)}", file=file)
    if uncategorized:
        print("  Uncategorized APIs:", file=file)
        for api in uncategorized:
            print(f"    - {api}", file=file)


def _diff(old_text, new_text, old_label, new_label):
    return ''.join(difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=old_label, tofile=new_label))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--provider', required=True, choices=sorted(PROVIDER_TAG))
    ap.add_argument('--taxonomy', required=True,
                    help='Path to lttng_categories.yaml')
    ap.add_argument('--tp-header', required=True,
                    help='Path to the generated rocm_<prov>_curated_tp.h')
    ap.add_argument('--out', required=True,
                    help='Path to write the JSON manifest')
    ap.add_argument('--check', action='store_true',
                    help='Do not write output. Generate to memory, diff '
                         'against the existing --out file, print a unified '
                         'diff and exit 1 on any mismatch.')
    ap.add_argument('--allow-uncategorized', action='store_true',
                    help='Do not fail when some curated APIs are not matched '
                         'by any category. By default an uncategorized API is '
                         'a hard error (a curated API silently outside every '
                         'category is almost always an omission).')
    args = ap.parse_args()

    with open(args.taxonomy) as f:
        taxonomy = yaml.safe_load(f)
    if not isinstance(taxonomy, dict):
        sys.exit(f"ERROR: taxonomy {args.taxonomy} must be a mapping of "
                 f"provider -> categories, got "
                 f"{type(taxonomy).__name__}")

    provider_tag = PROVIDER_TAG[args.provider]
    events = parse_tp_header(args.tp_header, provider_tag)
    try:
        manifest, by_base = build_manifest(
            args.provider, taxonomy, events,
            allow_uncategorized=args.allow_uncategorized)
    except TaxonomyError as e:
        sys.exit(f"ERROR: {e}")

    # Sanity check: every event name we're about to write actually exists
    # in the parsed tp.h event set (should be tautological given how
    # `categories` and `all_apis` are constructed, but cheap to assert).
    event_set = set(events)
    for cat_name, cat_events in manifest['categories'].items():
        for ev in cat_events:
            assert ev in event_set, (
                f"internal error: manifest event {ev!r} in category "
                f"{cat_name!r} not present in parsed tp.h event set")

    print_summary(args.provider, manifest, by_base)

    new_text = render(manifest)

    if args.check:
        old_text = ''
        if os.path.exists(args.out):
            with open(args.out) as f:
                old_text = f.read()
        if old_text != new_text:
            print(f"DRIFT: {args.out} does not match generator output:",
                  file=sys.stderr)
            print(_diff(old_text, new_text, args.out, f"{args.out} (generated)"),
                  file=sys.stderr)
            sys.exit(1)
        print(f"OK: {args.out} matches generator output", file=sys.stderr)
        return

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(new_text)
    print(f"wrote {args.out} ({len(new_text)} B)", file=sys.stderr)


if __name__ == '__main__':
    main()
