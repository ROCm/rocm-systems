# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse an OpenSpec directory into a data model.

Owns every filesystem and markdown-structure concern: discovering
capabilities and changes, parsing spec.md into Requirements and Scenarios,
reading the project context out of config.yaml, and the ``--check``
structural validation. Nothing here emits HTML.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

DELTA_KINDS = ("ADDED", "MODIFIED", "REMOVED", "RENAMED")
STEP_KINDS = ("GIVEN", "WHEN", "THEN", "AND", "BUT", "IF", "ELSE")


@dataclass
class Step:
    """One ``- **WHEN** ...`` bullet inside a scenario."""

    kind: str  # "WHEN", "THEN", ... or "" for a plain bullet
    text: str


@dataclass
class Scenario:
    title: str
    steps: List[Step] = field(default_factory=list)
    line: int = 0
    slug: str = ""


@dataclass
class Requirement:
    title: str
    body: List[str] = field(default_factory=list)
    scenarios: List[Scenario] = field(default_factory=list)
    line: int = 0
    slug: str = ""
    delta: str = ""  # ADDED / MODIFIED / REMOVED when this is a change delta


@dataclass
class Capability:
    cid: str
    title: str
    path: Path
    purpose: List[str] = field(default_factory=list)
    requirements: List[Requirement] = field(default_factory=list)
    raw: str = ""
    slug: str = ""
    #: id of the project this capability belongs to; set by :func:`load_site`
    project: str = ""
    #: True when this is a change delta rather than a baseline capability
    is_delta: bool = False
    #: capability ids this one references, already resolved against the loaded
    #: id set (bracketed ``[an-id]`` and backticked ``an-id`` forms both count)
    refs: List[str] = field(default_factory=list)

    @property
    def scenario_count(self) -> int:
        return sum(len(r.scenarios) for r in self.requirements)

    @property
    def delta_counts(self) -> "Dict[str, int]":
        """``{"ADDED": n, ...}`` over this capability's requirements.

        Empty for a baseline capability, populated for a change delta.
        """
        out: Dict[str, int] = {}
        for r in self.requirements:
            if r.delta:
                out[r.delta] = out.get(r.delta, 0) + 1
        return out


@dataclass
class Task:
    """One ``- [ ]`` / ``- [x]`` line in a change's tasks.md."""

    text: str
    done: bool = False
    depth: int = 0


@dataclass
class TaskPhase:
    """A ``## <heading>`` group of tasks, or the implicit ungrouped one."""

    name: str
    tasks: List[Task] = field(default_factory=list)
    slug: str = ""

    @property
    def done_count(self) -> int:
        return sum(1 for t in self.tasks if t.done)


@dataclass
class Change:
    cid: str
    path: Path
    #: ``[(document title, body lines)]`` for proposal.md, design.md and any
    #: other prose document in the change directory, in display order.
    docs: List[Tuple[str, List[str]]] = field(default_factory=list)
    #: delta specs; ``Capability.cid`` is the full path below ``specs/``, so a
    #: nested ``specs/cuid/identifier-format/spec.md`` yields
    #: ``cuid/identifier-format``.
    deltas: List[Capability] = field(default_factory=list)
    phases: List[TaskPhase] = field(default_factory=list)
    slug: str = ""
    title: str = ""
    project: str = ""
    #: capability ids the change's own prose references, resolved against the
    #: loaded id set. The CUID proposals name their capabilities here and
    #: nowhere else, so this -- not ``deltas[].refs`` -- carries that corpus's
    #: cross references.
    refs: List[str] = field(default_factory=list)

    @property
    def task_count(self) -> int:
        return sum(len(p.tasks) for p in self.phases)

    @property
    def task_done_count(self) -> int:
        return sum(p.done_count for p in self.phases)


@dataclass
class Project:
    name: str
    root: Path
    context: str
    capabilities: List[Capability]
    changes: List[Change]
    slug: str = ""
    #: canonical https URL of the code these specs describe, or "" when the
    #: corpus names none and sits in no git checkout. See :func:`parse_source`
    #: and :func:`git_source` for the two derivations.
    source: str = ""

    @property
    def requirement_count(self) -> int:
        return sum(len(c.requirements) for c in self.capabilities)

    @property
    def scenario_count(self) -> int:
        return sum(c.scenario_count for c in self.capabilities)

    @property
    def capability_ids(self) -> "Set[str]":
        return {c.cid for c in self.capabilities}


@dataclass
class Site:
    """One or more projects rendered into a single page."""

    projects: List[Project] = field(default_factory=list)
    #: reference token -> canonical capability id; built by :meth:`reindex`
    index: Dict[str, str] = field(default_factory=dict)

    @property
    def multi(self) -> bool:
        return len(self.projects) > 1

    def capability_owner(self, cid: str) -> Optional[str]:
        """Project slug that defines ``cid`` as a baseline capability."""
        for p in self.projects:
            if cid in p.capability_ids:
                return p.slug
        return None

    def baseline_ids(self) -> "Set[str]":
        out: Set[str] = set()
        for p in self.projects:
            out |= p.capability_ids
        return out

    def delta_ids(self) -> "List[str]":
        return [d.cid for p in self.projects for c in p.changes for d in c.deltas]

    def reindex(self) -> None:
        """(Re)build :attr:`index` from the currently loaded projects.

        A token resolves with baseline capabilities preferred over change
        deltas, and a whole id preferred over a trailing path segment, so that
        ``cuid/identifier-format`` -- which no baseline defines -- still finds
        the delta of that name.
        """
        ranked: Dict[str, Tuple[int, str]] = {}

        def offer(token: str, cid: str, prio: int) -> None:
            if token and (token not in ranked or prio < ranked[token][0]):
                ranked[token] = (prio, cid)

        for cid in sorted(self.baseline_ids()):
            offer(cid, cid, 0)
            offer(cid.rsplit("/", 1)[-1], cid, 1)
        for cid in sorted(set(self.delta_ids())):
            offer(cid, cid, 2)
            offer(cid.rsplit("/", 1)[-1], cid, 3)
        self.index = {token: cid for token, (_, cid) in ranked.items()}

    def resolve(self, cid: str) -> Optional[str]:
        """Baseline capability an id names, matched by last path segment.

        This is the cross-project link: a change delta id such as
        ``cuid/amdsmi-cli`` resolves to the ``amdsmi-cli`` capability the
        amdsmi project defines. An id no loaded project defines as a baseline
        capability -- ``cuid/amdsmi-identity-api``, say -- resolves to None.
        """
        baseline = self.baseline_ids()
        for key in (cid, cid.rsplit("/", 1)[-1]):
            if key in baseline:
                return key
        return None

    def resolve_ref(self, token: str) -> Optional[str]:
        """Canonical id of any loaded capability a reference token names.

        Unlike :meth:`resolve` this also finds change deltas, which is what a
        changes-only project needs; a token nothing defines is dropped.
        """
        if not self.index and self.projects:
            self.reindex()
        for key in (token, token.rsplit("/", 1)[-1]):
            hit = self.index.get(key)
            if hit:
                return hit
        return None


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------

RE_TITLE = re.compile(r"^#\s+(.*?)\s*$")
RE_H2 = re.compile(r"^##\s+(.*?)\s*$")
RE_REQ = re.compile(r"^###\s+Requirement:?\s*(.*?)\s*$", re.I)
RE_SCN = re.compile(r"^####\s+Scenario:?\s*(.*?)\s*$", re.I)
RE_FENCE = re.compile(r"^\s*(```|~~~)")
RE_STEP = re.compile(r"^\s*[-*+]\s+\*\*([A-Z][A-Z ]*)\*\*\s*(.*)$")
RE_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
RE_TASK = re.compile(r"^(\s*)[-*+]\s+\[([ xX])\]\s*(.*?)\s*$")

#: a capability-id shaped token: lowercase, dash separated, optionally nested
_CID = r"[a-z][a-z0-9]*(?:[-/][a-z0-9]+)+"
#: bare ``[an-id]`` cross reference (not a markdown link)
RE_XREF = re.compile(r"\[(" + _CID + r")\](?!\()")
#: ``an-id`` in backticks, the form the CUID corpus uses throughout
RE_CODEREF = re.compile(r"`(" + _CID + r")`")

_SLUG_BAD = re.compile(r"[^a-z0-9]+")


def slugify(*parts: str) -> str:
    joined = "-".join(p for p in parts if p)
    return _SLUG_BAD.sub("-", joined.lower()).strip("-")[:96] or "x"


def humanise(cid: str) -> str:
    """``add-cuid-kernel-interface`` -> ``Add cuid kernel interface``."""
    text = " ".join(w for w in re.split(r"[-_/\s]+", cid.strip()) if w)
    return (text[:1].upper() + text[1:]) if text else cid


def _strip_blank(lines: List[str]) -> List[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def parse_spec(path: Path, cid: str, is_delta: bool = False) -> Capability:
    """Parse one spec.md into a Capability.

    Tolerant by design: headings may vary in case and punctuation, sections may
    be missing, and Requirements are picked up whether or not they sit under a
    ``## Requirements`` (or ``## ADDED Requirements``) heading. A delta spec
    legitimately has no ``## Purpose``; that is the caller's business, not a
    parse error.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    cap = Capability(
        cid=cid, title=cid, path=path, raw=text, slug=slugify("cap", cid), is_delta=is_delta
    )

    sink: Optional[List[str]] = None
    req: Optional[Requirement] = None
    scn: Optional[Scenario] = None
    delta = ""
    fenced = False
    blank = False

    for no, line in enumerate(text.splitlines(), 1):
        if RE_FENCE.match(line):
            fenced = not fenced
        if not fenced:
            m = RE_TITLE.match(line)
            if m:
                cap.title = re.sub(r"\s+Specification$", "", m.group(1)).strip() or cid
                sink = None
                continue
            m = RE_H2.match(line)
            if m:
                head = m.group(1).strip()
                word = head.split(" ", 1)[0].upper()
                delta = word if word in DELTA_KINDS else ""
                sink = cap.purpose if head.lower().startswith("purpose") else None
                req = scn = None
                continue
            m = RE_REQ.match(line)
            if m:
                req = Requirement(title=m.group(1), line=no, delta=delta)
                req.slug = slugify("req", cid, req.title)
                cap.requirements.append(req)
                sink = req.body
                scn = None
                continue
            m = RE_SCN.match(line)
            if m:
                if req is None:  # scenario before any requirement: synthesize one
                    req = Requirement(title="(ungrouped)", line=no, delta=delta)
                    req.slug = slugify("req", cid, "ungrouped")
                    cap.requirements.append(req)
                scn = Scenario(title=m.group(1), line=no)
                scn.slug = slugify("scn", cid, req.title, scn.title)
                req.scenarios.append(scn)
                sink = None
                continue
        if scn is not None:
            _absorb_step(scn, line, blank)
        elif sink is not None:
            sink.append(line)
        blank = not line.strip()

    _dedupe_slugs(cap)
    _strip_blank(cap.purpose)
    for r in cap.requirements:
        _strip_blank(r.body)
    return cap


def _absorb_step(scn: Scenario, line: str, after_blank: bool = False) -> None:
    m = RE_STEP.match(line)
    if m:
        kind = m.group(1).strip().upper()
        scn.steps.append(Step(kind if kind in STEP_KINDS else "", m.group(2).strip()))
        return
    m = RE_BULLET.match(line)
    if m:
        scn.steps.append(Step("", m.group(1).strip()))
        return
    if not line.strip():
        return
    if scn.steps and not after_blank:  # wrapped continuation of the previous bullet
        scn.steps[-1].text += " " + line.strip()
    else:  # a trailing rationale paragraph: its own step, not glued onto a THEN
        scn.steps.append(Step("", line.strip()))


def _dedupe_slugs(cap: Capability) -> None:
    seen: Set[str] = set()

    def uniq(s: str) -> str:
        out, n = s, 2
        while out in seen:
            out, n = f"{s}-{n}", n + 1
        seen.add(out)
        return out

    cap.slug = uniq(cap.slug)
    for r in cap.requirements:
        r.slug = uniq(r.slug)
        for s in r.scenarios:
            s.slug = uniq(s.slug)


UNGROUPED = "(ungrouped)"


def parse_tasks(path: Path, cid: str = "") -> List[TaskPhase]:
    """Parse a change's tasks.md into phases of checkbox tasks.

    Tasks are ``- [ ]`` / ``- [x]`` bullets grouped by the ``##`` heading above
    them, with an implicit phase for tasks that precede the first heading.
    Nesting depth comes from the leading indentation, whatever width the file
    happens to use. A tasks.md of pure prose yields no phases at all.
    """
    if not path.is_file():
        return []
    phases: List[TaskPhase] = []
    phase: Optional[TaskPhase] = None
    task: Optional[Task] = None
    indents: List[int] = []  # open indentation levels, innermost last
    task_indent = 0
    fenced = False

    def open_phase(name: str) -> TaskPhase:
        new = TaskPhase(name=name, slug=slugify("phase", cid, name))
        phases.append(new)
        return new

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if RE_FENCE.match(line):
            fenced = not fenced
            task = None
            continue
        if not fenced:
            m = RE_H2.match(line)
            if m:
                phase = open_phase(m.group(1).strip())
                indents, task = [], None
                continue
            m = RE_TASK.match(line)
            if m:
                if phase is None:
                    phase = open_phase(UNGROUPED)
                task_indent = len(m.group(1).expandtabs(4))
                while indents and indents[-1] > task_indent:
                    indents.pop()
                if not indents or indents[-1] < task_indent:
                    indents.append(task_indent)
                task = Task(text=m.group(3), done=m.group(2).lower() == "x", depth=len(indents) - 1)
                phase.tasks.append(task)
                continue
        if not line.strip():
            task = None
        elif task is not None and len(line) - len(line.lstrip()) > task_indent:
            task.text += " " + line.strip()  # wrapped continuation of the task
        else:
            task = None
    return [p for p in phases if p.tasks]


def _read_doc(path: Path) -> Tuple[str, List[str]]:
    """``(title, body)`` for one prose document; a leading H1 becomes the title."""
    body = path.read_text(encoding="utf-8", errors="replace").splitlines()
    title = ""
    for i, line in enumerate(body):
        if not line.strip():
            continue
        m = RE_TITLE.match(line)
        if m and m.group(1).strip():
            title = m.group(1).strip()
            del body[: i + 1]
        break
    return title or humanise(path.stem), _strip_blank(body)


def _change_doc_paths(path: Path) -> List[Path]:
    """proposal.md, then design.md, then any other top-level .md but tasks.md."""
    lead = ["proposal.md", "design.md"]
    rest = sorted(p for p in path.glob("*.md") if p.name not in lead + ["tasks.md"])
    return [path / n for n in lead if (path / n).is_file()] + rest


def parse_change(path: Path) -> Change:
    """Parse openspec/changes/<id>/: proposal, design, tasks, and delta specs."""
    chg = Change(cid=path.name, path=path, slug=slugify("chg", path.name))
    for doc in _change_doc_paths(path):
        title, body = _read_doc(doc)
        if body:
            chg.docs.append((title, body))
        if doc.name == "proposal.md" and title != humanise(doc.stem):
            chg.title = title
    chg.title = chg.title or humanise(path.name)

    specs = path / "specs"
    for spec in sorted(specs.rglob("spec.md")):
        rel = spec.parent.relative_to(specs).parts
        chg.deltas.append(parse_spec(spec, "/".join(rel) or chg.cid, is_delta=True))

    chg.phases = parse_tasks(path / "tasks.md", chg.cid)
    return chg


def parse_context(config: Path) -> str:
    """Pull the ``context:`` block out of config.yaml without a yaml module.

    Handles the ``context: |`` block-scalar form the OpenSpec CLI writes, and a
    plain one-line ``context: text`` value.
    """
    if not config.is_file():
        return ""
    lines = config.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^context:\s*(\|[-+>]?|>[-+]?)?\s*(.*)$", line)
        if not m:
            continue
        if not m.group(1):
            return m.group(2).strip()
        body: List[str] = []
        for nxt in lines[i + 1 :]:
            if nxt.strip() and not nxt.startswith((" ", "\t")):
                break
            body.append(nxt)
        indent = min((len(x) - len(x.lstrip()) for x in body if x.strip()), default=0)
        return "\n".join(x[indent:] if len(x) >= indent else x for x in body).strip("\n")
    return ""


# --------------------------------------------------------------------------
# provenance: the code these specs describe
# --------------------------------------------------------------------------
#
# Two derivations, tried in order. A corpus may name its source outright in
# config.yaml, which is the escape hatch for one kept outside any checkout of
# the code it specifies; otherwise the surrounding git checkout is asked. A
# corpus that answers neither has no source, and the page simply says less.


def parse_source(config: Path) -> str:
    """The optional ``source:`` URL in config.yaml, read without a yaml module.

    A plain scalar at column zero, so an indented ``source:`` inside the
    ``context:`` block scalar -- or a commented-out one -- is not mistaken for
    the key itself.
    """
    if not config.is_file():
        return ""
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^source:\s*(.*?)\s*$", line)
        if m:
            return m.group(1).strip("'\"")
    return ""


#: ``git@host:org/repo.git`` and ``https://host/org/repo(.git)`` alike
RE_REMOTE = re.compile(r"^(?:ssh://)?(?:git@|https?://)([^/:]+)[:/]+(.+?)(?:\.git)?/*$")
#: a linked worktree's ``.git`` is a file holding this
RE_GITDIR = re.compile(r"^gitdir:\s*(.+?)\s*$", re.M)
#: ``[remote "origin"]``, whatever spacing the file uses
RE_ORIGIN = re.compile(r'^\[remote\s+"origin"\]')


def git_source(root: Path) -> str:
    """A ``/tree/<branch>/<project path>`` URL for the checkout ``root`` sits in.

    Empty when there is no checkout above ``root``, or it has no ``origin``.
    """
    found = _git_dir(root)
    if not found:
        return ""
    work, gitdir = found
    base = _github_url(_git_origin(_git_common(gitdir) / "config"))
    if not base:
        return ""
    branch = _git_branch(gitdir)
    if not branch:
        return base
    # the project is the directory holding openspec/, not openspec/ itself
    rel = _relative(root.parent, work)
    return f"{base}/tree/{branch}" + (f"/{rel}" if rel else "")


def source_org(url: str) -> str:
    """The organisation segment of a source URL: ``https://host/ORG/repo/...``."""
    m = re.match(r"^https?://[^/]+/([^/]+)", url)
    return m.group(1) if m else ""


def _git_dir(root: Path) -> Optional[Tuple[Path, Path]]:
    """``(work tree root, .git directory)`` for the checkout containing ``root``.

    Handles a linked worktree, where ``.git`` is a *file* holding
    ``gitdir: <path>`` rather than the directory itself.
    """
    for work in [root] + list(root.parents):
        dot = work / ".git"
        try:
            if dot.is_dir():
                return work, dot
            if dot.is_file():
                m = RE_GITDIR.search(dot.read_text(encoding="utf-8", errors="replace"))
                if m:
                    linked = Path(m.group(1)).expanduser()
                    if not linked.is_absolute():
                        linked = work / linked
                    if linked.is_dir():
                        return work, linked
        except OSError:
            return None
    return None


def _git_common(gitdir: Path) -> Path:
    """The shared .git directory: a worktree keeps config in the main one."""
    common = gitdir / "commondir"
    try:
        if common.is_file():
            rel = common.read_text(encoding="utf-8", errors="replace").strip()
            if rel:
                return (gitdir / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    except OSError:
        pass
    return gitdir


def _git_origin(config: Path) -> str:
    """The ``origin`` remote's url out of a git config, parsed as plain ini."""
    try:
        if not config.is_file():
            return ""
        lines = config.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    in_origin = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("["):
            in_origin = bool(RE_ORIGIN.match(line))
        elif in_origin:
            m = re.match(r"^url\s*=\s*(.+?)\s*$", line)
            if m:
                return m.group(1)
    return ""


def _git_branch(gitdir: Path) -> str:
    """The checked-out branch, or the raw commit when HEAD is detached."""
    try:
        head = (gitdir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    m = re.match(r"^ref:\s*refs/heads/(.+)$", head)
    return m.group(1) if m else head


def _github_url(remote: str) -> str:
    """``git@github.com:ORG/REPO.git`` -> ``https://github.com/ORG/REPO``."""
    m = RE_REMOTE.match(remote.strip())
    if not m or "/" not in m.group(2):
        return ""
    return "https://{}/{}".format(m.group(1), m.group(2))


def _relative(path: Path, base: Path) -> str:
    """``path`` below ``base`` as a posix string, or "" when it is not below."""
    try:
        rel = path.resolve().relative_to(base.resolve())
    except ValueError:
        return ""
    return "" if str(rel) == "." else rel.as_posix()


def project_slug(root: Path) -> str:
    """Stable id for a project: the name of the directory holding openspec/.

    Resolves first, so a relative root such as ``../openspec`` still yields
    ``amdsmi`` rather than the useless slug of ``..``.
    """
    root = Path(root).expanduser().resolve()
    return slugify(root.parent.name or root.name)


def load_project(root: Path) -> Project:
    root = Path(root).expanduser().resolve()
    specs = root / "specs"
    caps = [
        parse_spec(p, "/".join(p.parent.relative_to(specs).parts))
        for p in sorted(specs.rglob("spec.md"))
    ]
    changes = [
        parse_change(d)
        for d in sorted((root / "changes").glob("*"))
        if d.is_dir() and d.name != "archive"
    ]
    changes = [c for c in changes if c.docs or c.deltas or c.phases]
    project = Project(
        name=root.parent.name or root.name,
        root=root,
        context=parse_context(root / "config.yaml"),
        capabilities=caps,
        changes=changes,
        slug=project_slug(root),
        source=parse_source(root / "config.yaml") or git_source(root),
    )
    _tag_project(project, project.slug)
    _uniquify([project])
    _resolve_refs(Site([project]))
    return project


def load_site(roots: Sequence[Path]) -> Site:
    """Load one or more openspec roots into a single Site.

    Project slugs come from each root's parent directory name and are made
    unique if two roots collide. Slugs of capabilities, requirements,
    scenarios, changes and phases are made unique across the whole site, so
    they are safe to use as HTML ids; the first project keeps its own.
    """
    site = Site()
    taken: Set[str] = set()
    loaded: Set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root in loaded:  # the same directory named twice is one project
            continue
        loaded.add(root)
        project = load_project(root)
        slug, n = project.slug or "project", 2
        while slug in taken:
            slug, n = f"{project.slug or 'project'}-{n}", n + 1
        taken.add(slug)
        _tag_project(project, slug)
        site.projects.append(project)
    _uniquify(site.projects)
    _resolve_refs(site)
    return site


def _tag_project(project: Project, slug: str) -> None:
    project.slug = slug
    for cap in _all_capabilities(project):
        cap.project = slug
    for chg in project.changes:
        chg.project = slug


def _all_capabilities(project: Project) -> Iterable[Capability]:
    yield from project.capabilities
    for chg in project.changes:
        yield from chg.deltas


def _uniquify(projects: Sequence[Project]) -> None:
    """Make every anchor slug unique across the given projects, in place."""
    seen: Set[str] = set()

    def uniq(s: str) -> str:
        out, n = s, 2
        while out in seen:
            out, n = f"{s}-{n}", n + 1
        seen.add(out)
        return out

    def take(cap: Capability) -> None:
        cap.slug = uniq(cap.slug)
        for req in cap.requirements:
            req.slug = uniq(req.slug)
            for scn in req.scenarios:
                scn.slug = uniq(scn.slug)

    for project in projects:
        for cap in project.capabilities:
            take(cap)
        for chg in project.changes:
            chg.slug = uniq(chg.slug)
            for phase in chg.phases:
                phase.slug = uniq(phase.slug)
            for cap in chg.deltas:
                take(cap)


def _refs_in(site: Site, text: str, own: str = "") -> List[str]:
    """Resolved capability ids named in ``text`` as ``[an-id]`` or ``an-id``."""
    found: List[str] = []
    for token in RE_XREF.findall(text) + RE_CODEREF.findall(text):
        hit = site.resolve_ref(token)
        if hit and hit != own and hit not in found:
            found.append(hit)
    return sorted(found)


def _resolve_refs(site: Site) -> None:
    """Fill ``Capability.refs`` and ``Change.refs`` from the bracketed and
    backticked ids in the prose, dropping any that name nothing loaded."""
    site.reindex()
    for project in site.projects:
        for cap in _all_capabilities(project):
            cap.refs = _refs_in(site, cap.raw, cap.cid)
        for chg in project.changes:
            prose = "\n".join(line for _, body in chg.docs for line in body)
            chg.refs = _refs_in(site, prose)


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def _check_capability(cap: Capability, problems: List[str]) -> None:
    rel = cap.path
    # a delta states only what it changes, so it needs no '## Purpose'
    if not cap.is_delta and not any(x.strip() for x in cap.purpose):
        problems.append(f"{rel}:1: missing or empty '## Purpose' section")
    if not cap.requirements:
        problems.append(f"{rel}:1: no '### Requirement:' headings found")
    for req in cap.requirements:
        where = f"{rel}:{req.line}: Requirement: {req.title}"
        if cap.is_delta and not req.delta:
            problems.append(
                f"{where}\n    not under an '## ADDED/MODIFIED/REMOVED Requirements' heading"
            )
        if not any(x.strip() for x in req.body):
            problems.append(f"{where}\n    no normative prose under the heading")
        if not req.scenarios and req.delta != "REMOVED":
            problems.append(f"{where}\n    no '#### Scenario:' under this requirement")
        for scn in req.scenarios:
            kinds = {s.kind for s in scn.steps}
            if "WHEN" not in kinds or "THEN" not in kinds:
                problems.append(
                    f"{rel}:{scn.line}: Scenario: {scn.title}\n"
                    f"    missing a '- **WHEN**' / '- **THEN**' bullet pair"
                )


def _check_change(chg: Change, problems: List[str]) -> None:
    proposal = chg.path / "proposal.md"
    if not proposal.is_file():
        problems.append(f"{proposal}:1: change has no proposal.md")
    tasks = chg.path / "tasks.md"
    if tasks.is_file() and not chg.task_count:
        problems.append(f"{tasks}:1: no '- [ ]' or '- [x]' task lines found")
    for cap in chg.deltas:
        _check_capability(cap, problems)


def check(project: Project) -> int:
    """Report specs that break the OpenSpec structure. Returns a problem count."""
    problems: List[str] = []
    for cap in project.capabilities:
        _check_capability(cap, problems)
    for chg in project.changes:
        _check_change(chg, problems)

    for p in problems:
        print(p, file=sys.stderr)
    n_files = sum(1 for _ in _all_capabilities(project))
    scope = f"{n_files} spec file(s)"
    if project.changes:
        scope += f" and {len(project.changes)} change(s)"
    if problems:
        print(f"\n{len(problems)} problem(s) across {scope}.", file=sys.stderr)
    else:
        print(f"{scope} checked, 0 problems.")
    return len(problems)
