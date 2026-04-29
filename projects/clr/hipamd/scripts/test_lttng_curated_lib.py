"""Unit tests for lttng_curated_lib. Run from worktree root:
    python3 -m pytest projects/clr/hipamd/scripts/test_lttng_curated_lib.py -v
or:
    python3 projects/clr/hipamd/scripts/test_lttng_curated_lib.py
"""
import io
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lttng_curated_lib import (
    parse_yaml_text, validate_api, expanded_field_count,
    DSL_TYPES, ALLOWED_DIRS, ParseError, BudgetError, IN_ARG_KIND,
)

# ---- Schema parsing ----

def test_parses_minimal_api():
    apis = parse_yaml_text("""
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,    type: ptr,    dir: IN}
    - {name: src,    type: ptr,    dir: IN}
""")
    assert len(apis) == 1
    assert apis[0]['api'] == 'hipMemcpyAsync'
    assert apis[0]['category'] == 'memory'
    assert len(apis[0]['args']) == 2

def test_rejects_missing_required_top_field():
    try:
        parse_yaml_text("- api: foo\n  args: []\n")  # no category
    except ParseError as e:
        assert 'category' in str(e)
        return
    raise AssertionError("expected ParseError")

def test_rejects_unknown_type():
    try:
        parse_yaml_text("""
- api: foo
  category: memory
  args: [{name: x, type: gizmo, dir: IN}]
""")
    except ParseError as e:
        assert 'gizmo' in str(e)
        return
    raise AssertionError("expected ParseError")

def test_rejects_unknown_dir():
    try:
        parse_yaml_text("""
- api: foo
  category: memory
  args: [{name: x, type: uint32, dir: SIDEWAYS}]
""")
    except ParseError as e:
        assert 'SIDEWAYS' in str(e)
        return
    raise AssertionError("expected ParseError")

def test_rejects_inout_v1():
    """dir: INOUT is hard-error in v1."""
    try:
        parse_yaml_text("""
- api: foo
  category: memory
  args: [{name: x, type: uint32, dir: INOUT}]
""")
    except ParseError as e:
        assert 'INOUT' in str(e)
        return
    raise AssertionError("expected ParseError")

# ---- Field-budget calculation ----

def test_field_count_simple():
    api = {'api': 'foo', 'category': 'memory', 'args': [
        {'name': 'a', 'type': 'uint32', 'dir': 'IN'},
        {'name': 'b', 'type': 'ptr',    'dir': 'OUT'},
    ]}
    # 2 args, each 1 field  => 2 payload fields
    assert expanded_field_count(api) == 2

def test_field_count_dim3_expands_to_3():
    api = {'api': 'foo', 'category': 'kernel_launch', 'args': [
        {'name': 'g', 'type': 'dim3', 'dir': 'IN'},
    ]}
    assert expanded_field_count(api) == 3

def test_field_count_dim3_packed_is_1():
    api = {'api': 'foo', 'category': 'kernel_launch', 'args': [
        {'name': 'g', 'type': 'dim3_packed', 'dir': 'IN'},
    ]}
    assert expanded_field_count(api) == 1

def test_field_count_hipLaunchKernel_natural():
    """High-arity case: hipLaunchKernel natural = 10 payload (at budget)."""
    api = {'api': 'hipLaunchKernel', 'category': 'kernel_launch', 'args': [
        {'name': 'function_address', 'type': 'ptr',    'dir': 'IN'},
        {'name': 'numBlocks',        'type': 'dim3',   'dir': 'IN'},
        {'name': 'dimBlocks',        'type': 'dim3',   'dir': 'IN'},
        {'name': 'args',             'type': 'ptr',    'dir': 'IN'},
        {'name': 'sharedMemBytes',   'type': 'size',   'dir': 'IN'},
        {'name': 'stream',           'type': 'handle', 'dir': 'IN'},
    ]}
    # 1 + 3 + 3 + 1 + 1 + 1 = 10 payload (exactly at the 10-field budget;
    # without mitigation, an 11th captured arg would push it over).
    assert expanded_field_count(api) == 10

def test_field_count_hipLaunchKernel_packed():
    """With dim3_packed mitigation, payload = 6 (within budget)."""
    api = {'api': 'hipLaunchKernel', 'category': 'kernel_launch', 'args': [
        {'name': 'function_address', 'type': 'ptr',         'dir': 'IN'},
        {'name': 'numBlocks',        'type': 'dim3_packed', 'dir': 'IN'},
        {'name': 'dimBlocks',        'type': 'dim3_packed', 'dir': 'IN'},
        {'name': 'args',             'type': 'ptr',         'dir': 'IN'},
        {'name': 'sharedMemBytes',   'type': 'size',        'dir': 'IN'},
        {'name': 'stream',           'type': 'handle',      'dir': 'IN'},
    ]}
    assert expanded_field_count(api) == 6

# ---- Budget enforcement (validate_api) ----

def test_validate_under_budget_passes():
    api = {'api': 'foo', 'category': 'memory', 'args': [
        {'name': 'a', 'type': 'uint32', 'dir': 'IN'},
    ]}
    validate_api(api)  # no raise

def test_validate_over_budget_raises():
    """Validation aborts on >10 payload fields."""
    api = {'api': 'foo', 'category': 'kernel_launch', 'args': [
        {'name': f'a{i}', 'type': 'uint32', 'dir': 'IN'} for i in range(11)
    ]}
    try:
        validate_api(api)
    except BudgetError as e:
        assert '11' in str(e)
        assert 'foo' in str(e)
        return
    raise AssertionError("expected BudgetError")

# ---- IN_ARG_KIND helper ----

def test_in_arg_kind_classifies_correctly():
    assert IN_ARG_KIND({'dir': 'IN'})    == 'IN'
    assert IN_ARG_KIND({'dir': 'OUT'})   == 'OUT'
    # INOUT is rejected upstream by parse_yaml_text but the helper still
    # answers honestly if called directly:
    assert IN_ARG_KIND({'dir': 'INOUT'}) == 'INOUT'

if __name__ == '__main__':
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn) and not inspect.isclass(fn):
            try:
                fn()
                print(f'  ok  {name}')
            except Exception as e:
                print(f'  FAIL {name}: {e}')
                failures += 1
    print(f'\n{"PASS" if failures == 0 else "FAIL"}: {failures} failures')
    sys.exit(1 if failures else 0)
