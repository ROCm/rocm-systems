# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_check_default_behavior.py."""

from dataclasses import replace
from pathlib import Path

import pytest

import rccl_check_default_behavior as gate

NEW_PARAM = 'NCCL_PARAM(RMADisable, "RMA_DISABLE", 0);'
SRC = "projects/rccl/src/rma/rma.cc"


@pytest.fixture(autouse=True)
def _outside_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the check off the real job summary and pull request when pytest runs in CI."""
    for name in (
        "GITHUB_ACTIONS",
        "GITHUB_API_URL",
        "GITHUB_REF",
        "GITHUB_REPOSITORY",
        "GITHUB_STEP_SUMMARY",
        "GITHUB_TOKEN",
        "PR_BODY",
        "PR_TITLE",
    ):
        monkeypatch.delenv(name, raising=False)


def _pending(
    before: dict[str, str],
    after: dict[str, str],
    changelog_added: str = "",
    env_docs: str = "",
    *,
    internal: bool = False,
    known: tuple[str, ...] = (),
) -> list[gate.Finding]:
    return gate.evaluate(
        gate.findings_between(before, after),
        changelog_added=changelog_added,
        env_docs=env_docs,
        internal=internal,
        known_before=set(known),
        known_after=set(known),
    )


def test_a_new_param_needs_a_changelog_line_and_a_doc_entry() -> None:
    after = {SRC: f"int x;\n{NEW_PARAM}\n"}
    changelog = "* `NCCL_RMA_DISABLE` (default `0`): off"
    pending = _pending({}, after)
    assert [(item.kind, item.primary_env, item.places) for item in pending] == [
        ("added", "NCCL_RMA_DISABLE", (gate.Place(SRC, 2),))
    ]
    assert [(item.needs_changelog, item.needs_docs) for item in pending] == [(True, True)]
    assert [(item.needs_changelog, item.needs_docs) for item in _pending({}, after, changelog)] == [(False, True)]
    assert _pending({}, after, changelog, env_docs="``NCCL_RMA_DISABLE``\n") == []


def test_only_a_new_variable_needs_a_doc_entry() -> None:
    before = {SRC: f"{NEW_PARAM}\n"}
    retuned = {SRC: NEW_PARAM.replace(", 0)", ", 1)") + "\n"}
    assert [(item.kind, item.needs_docs) for item in _pending(before, retuned)] == [("default", False)]
    assert [(item.kind, item.needs_docs) for item in _pending(before, {SRC: "int x;\n"})] == [("removed", False)]


def test_internal_env_waives_the_doc_entry_but_not_the_changelog() -> None:
    pending = _pending({}, {SRC: f"{NEW_PARAM}\n"}, internal=True)
    assert [(item.needs_changelog, item.needs_docs) for item in pending] == [(True, False)]


def test_a_missing_changelog_line_is_an_error_and_a_missing_doc_entry_a_warning() -> None:
    finding = _pending({}, {SRC: f"{NEW_PARAM}\n"})[0]
    pages = f"{gate.ENV_DOCS[0]} or {gate.ENV_DOCS[1]}"
    error, warning = gate._annotations(finding)
    assert error.startswith(f"::error file={SRC},line=1::")
    assert f"Name it in {gate.CHANGELOG}, or put {gate.SKIP_TOKEN}" in error
    assert warning.startswith(f"::warning file={SRC},line=1::")
    assert f"Document it in {pages}, or put {gate.INTERNAL_TOKEN}" in warning
    assert gate._annotations(replace(finding, needs_changelog=False)) == [warning]


def test_default_edit_reports_old_and_new_values() -> None:
    path = "projects/rccl/src/init.cc"
    before = {path: 'RCCL_PARAM(CeAllreduce, "CE_ALLREDUCE", 0);\n'}
    after = {path: 'RCCL_PARAM(CeAllreduce, "CE_ALLREDUCE", -1);\n'}
    pending = _pending(before, after)
    assert len(pending) == 1
    finding = pending[0]
    assert finding.kind == "default"
    assert finding.primary_env == "RCCL_CE_ALLREDUCE"
    assert finding.old_default == "0"
    assert finding.new_default == "-1"
    assert "default changed `0` -> `-1`" in gate.suggestion(finding)


def test_multiline_default_edit_is_caught() -> None:
    path = "projects/rccl/src/misc/signals.cc"
    before = {path: 'RCCL_PARAM(EnableSignalHandler, "ENABLE_SIGNALHANDLER",\n           0);\n'}
    after = {path: 'RCCL_PARAM(EnableSignalHandler, "ENABLE_SIGNALHANDLER",\n           1);\n'}
    pending = _pending(before, after)
    assert len(pending) == 1
    assert pending[0].kind == "default"
    assert pending[0].places == (gate.Place(path, 1),)
    assert pending[0].old_default == "0"
    assert pending[0].new_default == "1"


def test_cast_and_block_comment_defaults_parse() -> None:
    source = "\n".join(
        [
            'NCCL_PARAM(SocketInlineSize, "SOCKET_INLINE", /*128 B=*/1 << 7);',
            'NCCL_PARAM(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB */',
            'RCCL_PARAM(DdaLLOneShotThreshold, "DDA_LL_ONESHOT_THRESHOLD", (size_t)(1) * 1024 * 1024); // 1 MiB',
            'NCCL_PARAM(LsaTeamSize, "LSA_TEAM_SIZE", 0)',
            'RCCL_PARAM(DdaThreshold, "DDA_THRESHOLD", kDdaThresholdUnset);',
            'RCCL_PARAM(CeArMaxMsgBytes, "CE_AR_MAX_MSG_BYTES", -1);  // -1 = from arch table (2-shot)',
        ]
    )
    calls = gate.calls_in(source, SRC)
    assert [call.default for call in calls] == [
        "1 << 7",
        "(1 << 17)",
        "(size_t)(1) * 1024 * 1024",
        "0",
        "kDdaThresholdUnset",
        "-1",
    ]


def test_comments_are_not_part_of_a_declaration() -> None:
    define = 'DEFINE_NCCL_PARAM(x, int, NCCL_FOO, // key\n                  0, 0, parser, "desc");\n'
    assert [(call.env_names, call.default) for call in gate.calls_in(define, SRC)] == [(("NCCL_FOO",), "0")]

    before = {SRC: 'NCCL_PARAM(X, "X", /* old note */ 0);\n'}
    after = {SRC: 'NCCL_PARAM(X, "X", /* new note */ 0);\n'}
    assert _pending(before, after) == []


def test_only_block_comment_lines_are_skipped() -> None:
    source = ' * Reads getenv("NCCL_DOC_ONLY") at init.\n*out = getenv("NCCL_NEW_KNOB");\n'
    assert [item.primary_env for item in _pending({}, {SRC: source})] == ["NCCL_NEW_KNOB"]


def test_renaming_the_c_identifier_is_not_a_behavior_change() -> None:
    before = {SRC: 'NCCL_PARAM(OldName, "FOO", 0);\n'}
    after = {SRC: 'NCCL_PARAM(NewName, "FOO", 0);\n'}
    assert _pending(before, after) == []


def test_moving_a_declaration_is_not_a_change_but_moving_it_with_a_new_default_is() -> None:
    other = "projects/rccl/src/rma/other.cc"
    before = {SRC: f"{NEW_PARAM}\n", other: "int y;\n"}
    assert _pending(before, {SRC: "int x;\n", other: f"int y;\n{NEW_PARAM}\n"}) == []

    moved = NEW_PARAM.replace(", 0)", ", 1)")
    pending = _pending(before, {SRC: "int x;\n", other: f"int y;\n{moved}\n"})
    assert [(item.kind, item.old_default, item.new_default, item.places) for item in pending] == [
        ("default", "0", "1", (gate.Place(other, 2),))
    ]


def test_removed_param_reports_only_its_old_default() -> None:
    before = {SRC: f'{NEW_PARAM}\nWARN("NCCL_RMA_DISABLE=%d", v);\n'}
    pending = _pending(before, {SRC: "int x;\n"})
    assert [(item.kind, item.primary_env, item.old_default, item.new_default) for item in pending] == [
        ("removed", "NCCL_RMA_DISABLE", "0", None)
    ]
    assert "| removed | `0` |  |" in gate._summary(pending, [])
    annotations = gate._annotations(pending[0])
    assert len(annotations) == 1
    assert annotations[0].startswith(f"::error file={SRC}::")


def test_log_text_alone_is_not_an_env_change() -> None:
    before = {SRC: 'WARN("QP sharing (NCCL_IB_QP_SHARING_ENABLE=1) is invalid");\n'}
    after = {SRC: 'WARN("QP sharing (RCCL_IB_QP_SHARING_ENABLE=1) is invalid");\n'}
    assert _pending(before, after) == []
    assert _pending({}, {SRC: 'INFO(NCCL_INIT, "type NCCL_GIN_CONNECTION_FULL");\n'}) == []


def test_a_new_variable_lists_its_log_mentions_too() -> None:
    other = "projects/rccl/src/rma/other.cc"
    after = {SRC: f"{NEW_PARAM}\n", other: 'int y;\nWARN("NCCL_RMA_DISABLE=%d is ignored", v);\n'}
    pending = _pending({}, after)
    assert [item.places for item in pending] == [(gate.Place(other, 2), gate.Place(SRC, 1))]


def test_alias_is_covered_by_either_prefix() -> None:
    after = {SRC: 'RCCL_PARAM_NCCL_ALIAS(IbCast, "IB_CAST", -1);\n'}
    assert _pending({}, after)[0].env_names == ("RCCL_IB_CAST", "NCCL_IB_CAST")
    assert _pending({}, after, "* NCCL_IB_CAST now aliases RCCL_IB_CAST", env_docs="``RCCL_IB_CAST``") == []


def test_adding_or_dropping_the_nccl_alias_adds_or_removes_that_name() -> None:
    plain = {SRC: 'RCCL_PARAM(IbCast, "IB_CAST", -1);\n'}
    alias = {SRC: 'RCCL_PARAM_NCCL_ALIAS(IbCast, "IB_CAST", -1);\n'}
    assert [(item.kind, item.primary_env) for item in _pending(plain, alias)] == [("added", "NCCL_IB_CAST")]
    assert [(item.kind, item.primary_env) for item in _pending(alias, plain)] == [("removed", "NCCL_IB_CAST")]


def test_a_longer_variable_name_does_not_cover_a_shorter_one() -> None:
    after = {SRC: 'NCCL_PARAM(Ib, "IB", 0);\n'}
    pending = _pending({}, after, "* `NCCL_IB_CAST` (default `0`): unrelated")
    assert [item.primary_env for item in pending] == ["NCCL_IB"]


def test_macro_definitions_are_ignored() -> None:
    header = "projects/rccl/src/include/param.h"
    source = "#define NCCL_PARAM(name, env, deftVal) \\\n"
    assert _pending({header: source}, {header: source + "\n"}) == []


def test_one_entry_per_variable_lists_every_file() -> None:
    call = 'const char* v = ncclGetEnv("NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME");\n'
    init = "projects/rccl/src/transport/net_ib_cast/init.cc"
    scheduler = "projects/rccl/src/transport/net_ib_cast/scheduler.cc"
    pending = _pending({}, {scheduler: call, init: call})
    assert [item.primary_env for item in pending] == ["NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME"]
    assert pending[0].places == (gate.Place(init, 1), gate.Place(scheduler, 1))

    summary = gate._summary(pending, [])
    assert f"| CHANGELOG, docs | `{init}:1`<br>`{scheduler}:1` |" in summary
    assert summary.count("* `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME`: <why>") == 1
    assert summary.count("* `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME`\n") == 1
    assert [line.split("::")[1] for line in gate._annotations(pending[0])] == [
        f"error file={init},line=1",
        f"error file={scheduler},line=1",
        f"warning file={init},line=1",
        f"warning file={scheduler},line=1",
    ]


def _in_pull_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ROCm/rocm-systems")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")


def _comment(body: str, url: str, author: str = gate.COMMENT_AUTHOR) -> dict[str, object]:
    return {"body": body, "url": url, "user": {"login": author}}


def _fake_github(monkeypatch: pytest.MonkeyPatch, comments: list[dict[str, object]]) -> list[tuple[str, str]]:
    """Serve ``comments`` as the PR thread, 100 per page. Return each (method, url) called."""
    _in_pull_request(monkeypatch)
    calls: list[tuple[str, str]] = []

    def request(method: str, url: str, token: str, payload: dict[str, str] | None) -> object:
        assert token == "token"
        calls.append((method, url))
        if method == "GET":
            page = int(url.rsplit("page=", 1)[1])
            return comments[(page - 1) * 100 : page * 100]
        assert payload is not None
        if method == "POST":
            comments.append(_comment(payload["body"], f"{url}/{len(comments)}"))
        else:
            next(item for item in comments if item["url"] == url)["body"] = payload["body"]
        return {}

    monkeypatch.setattr(gate, "_github_json", request)
    return calls


def test_failure_comment_is_created_then_edited_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    comments: list[dict[str, object]] = []
    calls = _fake_github(monkeypatch, comments)
    gate.sync_pull_request_comment("first failure\n", create=True)
    gate.sync_pull_request_comment("second failure\n", create=True)
    assert [method for method, _ in calls] == ["GET", "POST", "GET", "PATCH"]
    assert [item["body"] for item in comments] == [f"{gate.COMMENT_MARKER}\nsecond failure\n"]


def test_passing_check_edits_an_existing_comment_and_never_creates_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comments: list[dict[str, object]] = []
    calls = _fake_github(monkeypatch, comments)
    gate.sync_pull_request_comment(gate.PASS_MESSAGE, create=False)
    assert comments == []

    comments.append(_comment(f"{gate.COMMENT_MARKER}\nfailure", "https://api.github.com/c/1"))
    gate.sync_pull_request_comment(gate.PASS_MESSAGE, create=False)
    assert [method for method, _ in calls] == ["GET", "GET", "PATCH"]
    assert comments[0]["body"] == f"{gate.COMMENT_MARKER}\n{gate.PASS_MESSAGE}"


def test_comment_search_reads_every_page_and_skips_a_quoted_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marked = f"{gate.COMMENT_MARKER}\nfailure"
    comments = [_comment("lgtm", f"https://api.github.com/c/{n}", author="dev") for n in range(99)]
    comments.append(_comment(marked, "https://api.github.com/c/quoted", author="dev"))
    comments.append(_comment(marked, "https://api.github.com/c/bot"))
    calls = _fake_github(monkeypatch, comments)
    gate.sync_pull_request_comment("second failure\n", create=True)
    assert [method for method, _ in calls] == ["GET", "GET", "PATCH"]
    assert calls[-1] == ("PATCH", "https://api.github.com/c/bot")


def test_comment_errors_become_a_warning_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _in_pull_request(monkeypatch)

    def request(method: str, url: str, token: str, payload: dict[str, str] | None) -> object:
        raise ConnectionResetError("reset by peer")

    monkeypatch.setattr(gate, "_github_json", request)
    gate.sync_pull_request_comment("failure\n", create=True)
    assert "::warning::Could not update the pull request comment" in capsys.readouterr().out


def test_comment_needs_a_token_and_warns_in_actions_without_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def request(method: str, url: str, token: str, payload: dict[str, str] | None) -> object:
        raise AssertionError("no GitHub call without a token")

    monkeypatch.setattr(gate, "_github_json", request)
    gate.sync_pull_request_comment("failure\n", create=True)
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
    gate.sync_pull_request_comment("failure\n", create=True)
    assert "::warning::No GITHUB_TOKEN" in capsys.readouterr().out


def test_suggestion_for_a_new_variable() -> None:
    finding = _pending({}, {SRC: f"{NEW_PARAM}\n"})[0]
    assert gate.suggestion(finding) == "* `NCCL_RMA_DISABLE` (default `0`): <why>"


def test_new_getenv_and_define_param_are_env_changes() -> None:
    getenv_only = {SRC: 'const char* arch = getenv("RCCL_EP_FAKE_ARCH");\n'}
    added = _pending({}, getenv_only)
    assert [(item.primary_env, item.new_default) for item in added] == [("RCCL_EP_FAKE_ARCH", None)]
    assert gate.suggestion(added[0]) == "* `RCCL_EP_FAKE_ARCH`: <why>"

    define = (
        "DEFINE_NCCL_PARAM(ncclParamDebugLevel, ncclDebugLogLevel, NCCL_DEBUG, NCCL_LOG_NONE,\n"
        '                  0, parser, "level");\n'
    )
    edited = define.replace("NCCL_LOG_NONE", "NCCL_LOG_WARN")
    pending = _pending({SRC: define}, {SRC: edited})
    assert len(pending) == 1
    assert pending[0].primary_env == "NCCL_DEBUG"
    assert pending[0].kind == "default"
    assert pending[0].old_default == "NCCL_LOG_NONE"
    assert pending[0].new_default == "NCCL_LOG_WARN"


def test_a_second_read_of_an_existing_env_is_not_new() -> None:
    first = 'const char* proto = ncclGetEnv("NCCL_PROTO");\n'
    second = first + 'const char* again = getenv("NCCL_PROTO");\n'
    assert _pending({SRC: first}, {SRC: second}) == []
    assert _pending({}, {"projects/rccl/src/other.cc": second}, known=("NCCL_PROTO",)) == []


def test_a_bare_nccl_token_is_not_an_env_by_itself() -> None:
    before = {SRC: "int flags = NCCL_INIT;\n"}
    after = {SRC: "int flags = NCCL_INIT | NCCL_COLL;\n"}
    assert _pending(before, after) == []


DEBUG_CC = """\
DEFINE_NCCL_PARAM(ncclParamDebugLevel, ncclDebugLogLevel, NCCL_DEBUG, NCCL_LOG_NONE,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamOneOf<ncclDebugLogLevel>(makeOptions(
                    makeOption("WARN", NCCL_LOG_WARN, "Prints only messages indicating a fatal error."),
                    makeOption("ABORT", NCCL_LOG_ABORT, "")
                  )), "Set debug output level");

DEFINE_NCCL_PARAM(ncclParamDebugSubsys, uint64_t, NCCL_DEBUG_SUBSYS,
                  NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  (ncclParamBitsetOf<ncclDebugLogSubSys, uint64_t>(makeOptions(
                    makeOption("INIT", NCCL_INIT, "NCCL and comm initialization (included in default)")
                  ))), "Filter debug output by (comma-separated)");

DEFINE_NCCL_PARAM(ncclParamWarnEnableDebugInfo, bool, NCCL_WARN_ENABLE_DEBUG_INFO, false,
                  NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "If enabled, the debug level will be set to INFO after a WARN level debug message is logged.");

DEFINE_NCCL_PARAM(ncclParamDebugTimestampLevel, uint32_t, NCCL_DEBUG_TIMESTAMP_LEVELS, (1u << NCCL_LOG_WARN),
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT,
                  ncclParamBitsetOf<uint32_t>(
                    makeOptions(makeOption("ALL",
                                           (1u << NCCL_LOG_VERSION | 1u << NCCL_LOG_WARN),
                                           "on All messages"))),
                  "Set which log lines get a timestamp depending upon the level of the log");

DEFINE_NCCL_PARAM(ncclParamDebugTsFormat, const char*, NCCL_DEBUG_TIMESTAMP_FORMAT, "[%F %T] ",
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "Set the format used when printing debug log messages");

DEFINE_NCCL_PARAM(ncclParamDebugFile, const char*, NCCL_DEBUG_FILE, nullptr,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, NCCL_PARAM_DEFAULT,
                  "Set the NCCL debug logging output to a file. "
                  "please convert to a relative or absolute path first.");
"""


def test_define_nccl_param_forms_from_debug_cc_parse() -> None:
    defaults = {
        call.env_names[0]: call.default
        for call in gate.calls_in(DEBUG_CC, "projects/rccl/src/debug.cc")
        if call.default is not None
    }
    assert defaults == {
        "NCCL_DEBUG": "NCCL_LOG_NONE",
        "NCCL_DEBUG_SUBSYS": "NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV",
        "NCCL_WARN_ENABLE_DEBUG_INFO": "false",
        "NCCL_DEBUG_TIMESTAMP_LEVELS": "(1u << NCCL_LOG_WARN)",
        "NCCL_DEBUG_TIMESTAMP_FORMAT": '"[%F %T] "',
        "NCCL_DEBUG_FILE": "nullptr",
    }


def test_every_declaration_in_the_tree_parses() -> None:
    repo = Path(__file__).resolve().parents[3]
    if not (repo / gate.SRC_TREE).is_dir():
        pytest.skip(f"{gate.SRC_TREE} is not in this checkout")
    paths = gate._git(repo, ["ls-files", gate.SRC_TREE]).splitlines()
    sources = {rel: (repo / rel).read_text(encoding="utf-8", errors="replace") for rel in paths}
    assert gate.unreadable_declarations(sources) == []


def _commit(repo: Path, files: dict[str, str], message: str) -> None:
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    gate._git(repo, ["add", "-A"])
    gate._git(repo, ["commit", "-m", message])


def _repo(tmp_path: Path) -> Path:
    gate._git(tmp_path, ["init", "-b", "main"])
    gate._git(tmp_path, ["config", "user.email", "rccl-ci@example.com"])
    gate._git(tmp_path, ["config", "user.name", "RCCL CI"])
    pages = {gate.ENV_DOCS[0]: "Environment Variables\n", gate.ENV_DOCS[1]: "RCCL environment variables\n"}
    _commit(tmp_path, {gate.CHANGELOG: "# Changelog\n", SRC: "int x;\n", **pages}, "base")
    return tmp_path


def test_known_names_cover_reads_and_unprefixed_declarations(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    source = (
        'getenv("NCCL_IB_CAST");\n'
        'NCCL_PARAM(IbMatching, "IB_MATCHING", -2);\n'
        'WARN("NCCL_IB_TEXT_ONLY=%d", v);\n'
    )
    _commit(repo, {"projects/rccl/src/ib.cc": source}, "ib")
    asked = {"NCCL_IB", "NCCL_IB_CAST", "NCCL_IB_MATCHING", "RCCL_IB_MATCHING", "NCCL_IB_TEXT_ONLY", "NCCL_ABSENT"}
    assert gate.known_names(repo, "HEAD", asked) == {"NCCL_IB_CAST", "NCCL_IB_MATCHING"}
    assert gate.known_names(repo, "HEAD^", asked) == set()


def test_check_fails_without_a_changelog_entry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {SRC: f"int x;\n{NEW_PARAM}\n"}, "add RMA disable")

    assert gate.check(repo, "HEAD^", []) == 1
    captured = capsys.readouterr()
    assert f"::error file={SRC},line=2::" in captured.out
    assert "NCCL_RMA_DISABLE" in captured.out


def test_check_passes_when_the_changelog_and_either_env_page_name_it(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    fast = 'RCCL_PARAM(Fast, "FAST", 1);'
    changelog = "# Changelog\n\n* `NCCL_RMA_DISABLE` (default `0`): off\n* `RCCL_FAST` (default `1`): on\n"
    _commit(
        repo,
        {
            SRC: f"int x;\n{NEW_PARAM}\n{fast}\n",
            gate.CHANGELOG: changelog,
            gate.ENV_DOCS[0]: "NCCL_RMA_DISABLE\n----------------\n",
            gate.ENV_DOCS[1]: "    * - | ``RCCL_FAST``\n",
        },
        "add RMA disable and fast",
    )

    assert gate.check(repo, "HEAD^", []) == 0


def test_a_missing_doc_entry_warns_and_comments_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    changelog = "# Changelog\n\n* `NCCL_RMA_DISABLE` (default `0`): off\n"
    _commit(repo, {SRC: f"int x;\n{NEW_PARAM}\n", gate.CHANGELOG: changelog}, "add RMA disable")
    comments: list[dict[str, object]] = []
    _fake_github(monkeypatch, comments)

    assert gate.check(repo, "HEAD^", []) == 0
    out = capsys.readouterr().out
    assert f"::warning file={SRC},line=2::NCCL_RMA_DISABLE added with default 0. Document it in" in out
    assert "::error" not in out
    assert len(comments) == 1
    assert "### New RCCL env vars without docs" in str(comments[0]["body"])
    assert "Not blocking" in str(comments[0]["body"])

    assert gate.check(repo, "HEAD^", [f"debug knob {gate.INTERNAL_TOKEN}"]) == 0
    assert "::warning" not in capsys.readouterr().out
    assert comments[0]["body"] == f"{gate.COMMENT_MARKER}\n{gate.PASS_MESSAGE}"


def test_check_passes_when_the_commit_subject_records_the_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {SRC: f"int x;\n{NEW_PARAM}\n"}, "reformat params [no-behavior-change]")

    assert gate.check(repo, "HEAD^", []) == 0
    assert gate.SKIPPED_MESSAGE in capsys.readouterr().out


def test_main_takes_the_override_from_the_pr_body_and_reports_git_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {SRC: f"int x;\n{NEW_PARAM}\n"}, "add RMA disable")
    monkeypatch.chdir(repo)

    assert gate.main([]) == 1
    monkeypatch.setenv("PR_BODY", f"retune only {gate.SKIP_TOKEN}")
    assert gate.main([]) == 0
    capsys.readouterr()
    assert gate.main(["--base-ref", "no-such-ref"]) == 1
    assert "no-such-ref" in capsys.readouterr().err


def test_check_fails_on_an_unreadable_declaration_in_a_changed_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {SRC: 'int x;\nNCCL_PARAM(Broken, "BROKEN");\n'}, "broken declaration")

    assert gate.check(repo, "HEAD^", []) == 1
    out = capsys.readouterr().out
    assert f"::error file={SRC},line=2::{gate.UNREADABLE_MESSAGE}" in out
    assert f"* `{SRC}:2`" in out
    assert gate.check(repo, "HEAD^", [f"cleanup {gate.SKIP_TOKEN}"]) == 0

    _commit(repo, {"projects/rccl/src/other.cc": "int y;\n"}, "unrelated")
    assert gate.check(repo, "HEAD^", []) == 0


def test_check_scans_only_product_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    mention = 'getenv("RCCL_ENABLE_HOST_GRAPH");\n'
    _commit(
        repo,
        {"projects/rccl/test/host/fakes.cc": mention, "projects/rccl/tools/scripts/run.py": mention},
        "tests and tools",
    )

    assert gate.check(repo, "HEAD^", []) == 0
    assert gate.PASS_MESSAGE in capsys.readouterr().out


def test_check_accepts_a_new_read_of_a_declared_variable(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {"projects/rccl/src/p2p.cc": 'NCCL_PARAM(IbMatching, "IB_MATCHING", -2);\n'}, "declare")
    _commit(repo, {"projects/rccl/src/init.cc": 'const char* v = getenv("NCCL_IB_MATCHING");\n'}, "read")

    assert gate.check(repo, "HEAD^", []) == 0


def test_check_reports_a_new_default_in_a_renamed_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    old, new = "projects/rccl/src/old.cc", "projects/rccl/src/new.cc"
    body = "".join(f"int filler{n};\n" for n in range(20))
    _commit(repo, {old: f"{body}{NEW_PARAM}\n"}, "add param")
    (repo / old).unlink()
    _commit(repo, {new: f"{body}{NEW_PARAM.replace(', 0)', ', 1)')}\n"}, "rename and retune")

    assert gate.check(repo, "HEAD^", []) == 1
    assert f"::error file={new},line=21::NCCL_RMA_DISABLE default changed 0 -> 1." in capsys.readouterr().out
