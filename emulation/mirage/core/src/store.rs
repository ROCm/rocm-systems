//! The on-disk configuration store: profiles, topologies and agents.
//!
//! These are user-authored documents, not session state. They live in the
//! config directory, they outlive every process, and reading one involves
//! nothing but the filesystem.
//!
//! That last point is why this module exists separately from the
//! supervisor. Every command answers configuration queries from here
//! directly, so `mirage profile list` is a directory read and nothing
//! else — there is nothing to start first. There is no cache to keep
//! coherent either: a `mirage run` reads a profile off disk when it
//! creates its session, so a profile written a moment earlier in another
//! terminal is already visible to it.
//!
//! A *write* asks two more questions, below.
//!
//! # Writes never destroy the user's work
//!
//! These files are the user's, and this module is the one door every
//! writer goes through, so the rule lives here rather than in each
//! command. Three things follow from it, and [`shipped`] — the set of
//! documents mirage itself seeds into a fresh config directory — is what
//! tells them apart:
//!
//! * A write that would replace a document mirage did not write is
//!   refused ([`profile_put`], [`topology_put`], [`agent_put`] alike —
//!   the three resource verbs are parallel, and a user has no way to
//!   infer that one of them destroys their work where the others refuse).
//!   Replacing a *pristine* builtin is fine: it is mirage's own seed and
//!   identical to the copy still compiled into the binary — but the
//!   caller is told it happened ([`Stored`]), because a builtin
//!   disappearing without a word is the same defect one level down.
//!   The name is claimed from the filesystem rather than from a stat, so
//!   two `create`s racing for it produce one winner and one refusal
//!   instead of two successes (see `write_new`).
//! * A delete of a pristine builtin is refused, because mirage rewrites
//!   every missing builtin on the next command — the file would come
//!   straight back and the "deleted" would have been a lie. Deleting one
//!   the user *has* changed is allowed, and is how a customised builtin
//!   is reset to the shipped version.
//! * A profile is checked before it lands, not when a session later tries
//!   to use it: its name, the topology and agent references inside it,
//!   and whether its emulator backend will accept it at all. A topology
//!   is checked the same way, for the same reason — see [`topology_put`].
//!
//! # A delete says what it broke
//!
//! Nothing here refuses a delete because another document refers to the
//! victim: the files are the user's. But a reference that stops resolving
//! is a failure they will meet later, in a command that has no visible
//! connection to the delete, so [`referrers_to`] names the documents that
//! just lost their target and the delete says so on the way out.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

use crate::agent::AgentDef;
use crate::common::MaybeRef;
use crate::error::{MirageError, Result};
use crate::profile::ProfileDef;
use crate::topology::TopologyDef;

/// One of the three kinds of document in the configuration store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DocKind {
    /// A [`ProfileDef`], under `<config>/profile/`.
    Profile,
    /// A [`TopologyDef`], under `<config>/topology/`.
    Topology,
    /// An [`AgentDef`], under `<config>/agent/`.
    Agent,
}

impl DocKind {
    /// The word this kind is called by on the command line and in
    /// messages (`mirage profile …`, "profile not found").
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            DocKind::Profile => "profile",
            DocKind::Topology => "topology",
            DocKind::Agent => "agent",
        }
    }

    /// Where a document of this kind called `name` lives.
    #[must_use]
    pub fn path(self, name: &str) -> PathBuf {
        match self {
            DocKind::Profile => crate::paths::profile_path(name),
            DocKind::Topology => crate::paths::topology_path(name),
            DocKind::Agent => crate::paths::agent_path(name),
        }
    }

    /// The directory every document of this kind lives in.
    ///
    /// The answer to "where did mirage look?", which is what a
    /// not-found error owes its reader; see [`MirageError::not_found`].
    #[must_use]
    pub fn root(self) -> PathBuf {
        match self {
            DocKind::Profile => crate::paths::profile_root(),
            DocKind::Topology => crate::paths::topology_root(),
            DocKind::Agent => crate::paths::agent_root(),
        }
    }

    /// The single spelling of `name` that this kind is addressed by.
    ///
    /// Profiles and agents are case-insensitive and stored lowercase;
    /// topologies are stored verbatim (`MI350X-1x8` is a topology name,
    /// not a GPU name, and the case is part of how it reads).
    ///
    /// Public because a *reference* to a document has to be written in
    /// the same spelling however it was obtained. `mirage profile create`
    /// took its agent from a flag on one path and from a picker reading
    /// this directory on the other, and stored `MI350X` or `mi350x`
    /// depending on which — the same profile, described two ways.
    #[must_use]
    pub fn canonical(self, name: &str) -> String {
        match self {
            DocKind::Profile | DocKind::Agent => name.to_lowercase(),
            DocKind::Topology => name.to_string(),
        }
    }
}

/// What storing a document did to the name it was given.
///
/// Returned rather than discarded because the two outcomes are not the
/// same news: one takes a free name, the other consumes mirage's own
/// shipped definition, and a user who did the second by accident has no
/// way to notice unless they are told.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stored {
    /// The name was free, and now holds this document.
    Created,
    /// The name held an untouched builtin, which this replaced. Nothing
    /// the user wrote was lost — the replaced copy is still compiled into
    /// the binary — but the shipped definition is no longer on disk, and
    /// deleting this document is what brings it back.
    ReplacedBuiltin,
}

/// The documents a linked-in provider seeds into a fresh config
/// directory, and re-seeds whenever one goes missing.
///
/// `mirage_builtin` submits exactly one of these; the core cannot depend
/// on it (the dependency runs the other way), so the list arrives at link
/// time the same way emulator backends do. A build without the provider
/// simply has no builtins, and every document on disk is then the user's.
#[derive(Debug)]
pub struct BuiltinDocuments {
    /// Every document the provider ships, as its kind, its name, and its
    /// content in the same shape [`crate::state::write_json`] writes.
    pub documents: fn() -> Vec<(DocKind, String, serde_json::Value)>,
}

inventory::collect!(BuiltinDocuments);

/// The content mirage ships for `name`, if it ships one at all.
///
/// Compared as parsed JSON rather than as bytes: what matters is whether
/// the document *says* something different from the shipped one, not
/// whether it was pretty-printed by the same version of serde.
#[must_use]
pub fn shipped(kind: DocKind, name: &str) -> Option<&'static serde_json::Value> {
    static INDEX: OnceLock<BTreeMap<(DocKind, String), serde_json::Value>> = OnceLock::new();
    let index = INDEX.get_or_init(|| {
        let mut out = BTreeMap::new();
        for provider in inventory::iter::<BuiltinDocuments> {
            for (kind, name, value) in (provider.documents)() {
                out.insert((kind, kind.canonical(&name)), value);
            }
        }
        out
    });
    index.get(&(kind, kind.canonical(name)))
}

/// Whether the document on disk is byte-for-meaning the one mirage
/// ships — so replacing or removing it destroys nothing the user wrote.
///
/// A document that cannot be read or parsed counts as *not* pristine:
/// whatever is there, mirage did not put it there in that state, and the
/// safe reading of "I cannot tell" is "do not touch it".
#[must_use]
pub fn is_pristine_builtin(kind: DocKind, name: &str) -> bool {
    let Some(shipped) = shipped(kind, name) else {
        return false;
    };
    let path = kind.path(name);
    let Ok(bytes) = std::fs::read(&path) else {
        return false;
    };
    matches!(serde_json::from_slice::<serde_json::Value>(&bytes), Ok(on_disk) if &on_disk == shipped)
}

/// The refusal to replace a document that mirage did not write.
///
/// Overwriting on request is fine; doing it without saying so is not.
/// `mirage profile create cdna4` on a name that is already taken used to
/// silently discard whatever was there, which for a profile someone had
/// tuned is unrecoverable — these files are the only copy.
fn already_exists(kind: DocKind, name: &str, path: &Path) -> MirageError {
    let kind = kind.as_str();
    let quoted = shown(name);
    MirageError::other(format!(
        "{kind} {quoted} already exists at {}, and mirage will not overwrite it. \
         Delete it first (`mirage {kind} delete {name}`) or choose another name.",
        path.display()
    ))
}

/// Store a document under `name`, refusing to destroy anything the user
/// wrote and saying so when it takes over a builtin.
///
/// The one door every `create` and `import` goes through, so the rule
/// lives here rather than three times over in the resource verbs.
fn put_document<T: Serialize>(kind: DocKind, name: &str, value: &T) -> Result<Stored> {
    let path = kind.path(name);
    // Mirage's own untouched seed is not the user's work, so taking the
    // name over is allowed — that is how a builtin gets customised. This
    // is the one write that replaces a file, so it goes the ordinary
    // rename way; the exclusive claim below exists to refuse exactly that.
    if is_pristine_builtin(kind, name) {
        crate::state::write_json(&path, value)?;
        return Ok(Stored::ReplacedBuiltin);
    }
    if path.exists() {
        return Err(already_exists(kind, name, &path));
    }
    write_new(kind, name, &path, value)?;
    Ok(Stored::Created)
}

/// Scratch files get a name no other writer can be using.
static SCRATCH: AtomicU64 = AtomicU64::new(0);

/// Write a document whose name must still be free when the bytes land.
///
/// The existence check above reads the directory and this writes to it,
/// and another `mirage … create` of the same name fits between the two:
/// both processes saw a free name, both printed `created`, and the
/// document left on disk was whichever one finished last — so the process
/// told it had created the profile had not created the profile that
/// exists. The exclusion is therefore taken from the filesystem instead
/// of inferred from a stat.
///
/// The bytes go to a scratch file first, as [`crate::state::write_json`]
/// does, so a concurrent reader never sees half a document; the name is
/// then claimed with `link`, which fails rather than replaces when the
/// target is taken. A plain `create_new` on the final path would be
/// exclusive too, but it would publish the name before the content, and
/// a scratch file with a fixed name would leave a crashed writer blocking
/// the name forever.
fn write_new<T: Serialize>(kind: DocKind, name: &str, path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| MirageError::io(parent, e))?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| MirageError::Json {
        path: path.to_path_buf(),
        source: e,
    })?;
    let scratch = path.with_extension(format!(
        "new.{}.{}",
        std::process::id(),
        SCRATCH.fetch_add(1, Ordering::Relaxed)
    ));
    // Every failure below names `path`, never the scratch file: the
    // scratch name is mirage's business, and a user told that
    // `mi350x.new.4711.0` could not be written has been handed a filename
    // that does not exist by the time they go looking for it.
    std::fs::write(&scratch, &bytes).map_err(|e| MirageError::io(path, e))?;
    let claimed = std::fs::hard_link(&scratch, path);
    // The scratch file has served its purpose either way, and leaving one
    // behind would put a stray `.new.<pid>.<n>` in a directory the user
    // reads and edits by hand.
    let _ = std::fs::remove_file(&scratch);
    claimed.map_err(|e| match e.kind() {
        std::io::ErrorKind::AlreadyExists => already_exists(kind, name, path),
        _ => MirageError::io(path, e),
    })
}

/// A name as an error may quote it back.
///
/// Echoing the argument verbatim turns a 5000-character name into a
/// 5000-character error line, which scrolls the sentence explaining the
/// problem off the screen. Long names are cut and counted instead; the
/// prefix is what a user recognises theirs by, and the count is what tells
/// them the shell expanded something they did not mean to pass.
fn shown(name: &str) -> String {
    const LIMIT: usize = 48;
    let mut kept: String = name.chars().take(LIMIT).collect();
    if kept.chars().count() < name.chars().count() {
        kept.push('…');
        return format!("{kept:?} ({} characters)", name.chars().count());
    }
    format!("{kept:?}")
}

/// Refuse a delete that would not stay deleted.
///
/// Mirage writes any missing builtin back on the next command, so
/// removing an untouched one changes nothing at all — and reporting
/// success for that is the defect, not the rewriting. A builtin the user
/// *has* edited is a different document that happens to share the name:
/// deleting it really does remove their version, and the shipped one
/// reappears in its place, so that delete is allowed and is the way to
/// reset a customised builtin.
fn guard_delete(kind: DocKind, name: &str) -> Result<()> {
    if !is_pristine_builtin(kind, name) {
        return Ok(());
    }
    let path = kind.path(name);
    let kind = kind.as_str();
    Err(MirageError::other(format!(
        "{kind} {name:?} is a builtin: mirage ships it and writes any missing builtin \
         back on the next command, so deleting it would report success and change \
         nothing. Edit {} instead — mirage never overwrites a builtin you have \
         changed, and deleting it once it differs does remove your version and \
         restore the shipped one.",
        path.display()
    )))
}

/// One document that names another.
///
/// Kept as a kind and a name rather than flattened to a string because
/// the two are what a user needs to go and fix it: which command edits
/// it, and which argument to give that command.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reference {
    /// The kind of document holding the reference.
    pub kind: DocKind,
    /// Its name, spelled as the command line addresses it.
    pub name: String,
}

impl std::fmt::Display for Reference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.kind.as_str(), self.name)
    }
}

/// Every document on disk that refers to `target` by name.
///
/// Nothing refers to a profile, so asking about one is an empty answer
/// rather than a mistake.
///
/// A document that cannot be read or parsed is passed over. It is not
/// evidence of a reference — mirage cannot see one in it — and treating
/// "unreadable" as "referring" would attach a warning about somebody's
/// broken file to an unrelated delete.
///
/// The comparison is per kind: an agent named `MI350X` and one named
/// `mi350x` are the same document and a topology may spell the reference
/// either way, where topology names are stored verbatim and are not.
#[must_use]
pub fn referrers_to(target: DocKind, name: &str) -> Vec<Reference> {
    let wanted = target.canonical(name);
    let same = |reference: &str| target.canonical(reference) == wanted;
    let mut found = Vec::new();

    if target == DocKind::Agent {
        for topology in topology_list().unwrap_or_default() {
            let Ok(def) =
                crate::state::read_json::<TopologyDef>(&DocKind::Topology.path(&topology))
            else {
                continue;
            };
            if matches!(&def.agent, MaybeRef::Ref(agent) if same(agent)) {
                found.push(Reference {
                    kind: DocKind::Topology,
                    name: topology,
                });
            }
        }
    }

    if matches!(target, DocKind::Agent | DocKind::Topology) {
        for profile in profile_list().unwrap_or_default() {
            let Ok(def) = crate::state::read_json::<ProfileDef>(&DocKind::Profile.path(&profile))
            else {
                continue;
            };
            // A profile reaches an agent through the topology it carries
            // inline, and reaches a topology directly. Both are the same
            // breakage to whoever runs the profile afterwards.
            let refers = match (&def.emulator.topology, target) {
                (MaybeRef::Ref(topology), DocKind::Topology) => same(topology),
                (MaybeRef::Owned(topology), DocKind::Agent) => {
                    matches!(&topology.agent, MaybeRef::Ref(agent) if same(agent))
                }
                _ => false,
            };
            if refers {
                found.push(Reference {
                    kind: DocKind::Profile,
                    name: profile,
                });
            }
        }
    }

    found.sort();
    found
}

/// Say which documents a delete has just broken.
///
/// The delete itself is allowed: these files are the user's and mirage
/// does not get a veto over which of them exist. What it may not do is
/// stay silent, which is what it did — `deleted agent mi350x` exited 0
/// and left every topology naming that agent failing at the next command
/// with an error about a reference the user had no reason to connect to
/// the delete they had just run.
///
/// A warning rather than a refusal, and after the fact rather than
/// before: deleting a document and then repointing what referred to it is
/// an ordinary way to work, and a prompt in the middle of it would be
/// friction for the case that is not a mistake.
fn warn_broken_referrers(kind: DocKind, name: &str, referrers: &[Reference]) {
    if referrers.is_empty() {
        return;
    }
    let named: Vec<String> = referrers.iter().map(Reference::to_string).collect();
    tracing::warn!(
        "deleting the {kind} {name} leaves {} referring to a {kind} that is no longer \
         there: {}. Each will fail wherever that reference is followed until it is \
         repointed or a {kind} of that name exists again.",
        if referrers.len() == 1 {
            "one document".to_string()
        } else {
            format!("{} documents", referrers.len())
        },
        named.join(", "),
        kind = kind.as_str(),
    );
}

/// Reject a document name that would escape its directory.
///
/// Every path here is built by interpolation —
/// `<config>/profile/<name>.json` — so a name is a path fragment under
/// another name. `..` in it walks out of the config directory, and a
/// leading `/` replaces it outright: `profile_get("../../../etc/passwd")`
/// reads an arbitrary file, and `profile_delete` with the same argument
/// deletes one.
///
/// That was survivable while these were CLI arguments the user typed
/// about their own machine. It is not now: a live `mirage run` serves
/// these over its socket, so the name can arrive off the wire, and every
/// other id that does (`SessionId`, `ExecId`) is validated at the serde
/// boundary. This is the same guarantee for the three that are plain
/// `String`s.
///
/// The same rule applies to a *reference* — the agent a topology names,
/// the topology a profile names — because a reference is interpolated
/// into a path by exactly the same rule when it is followed. See
/// [`validate_profile_refs`]; the agent and topology stores call this on
/// the way in, so a reference is checked wherever it is resolved rather
/// than only where a user typed one.
///
/// # Errors
///
/// Returns [`MirageError::Id`]-shaped rejection describing what is wrong.
pub fn validate_name(kind: DocKind, name: &str) -> Result<()> {
    let kind = kind.as_str();
    let bad = |why: &str| {
        Err(MirageError::other(format!(
            "invalid {kind} name {}: {why}",
            shown(name)
        )))
    };
    if name.is_empty() {
        return bad("it is empty");
    }
    if name.len() > 128 {
        return bad("it is longer than 128 characters");
    }
    if name.starts_with('.') {
        return bad("it starts with '.'");
    }
    // The property that matters is that the name stays a single path
    // component: no separator, no parent reference, nothing the
    // filesystem reads as structure. Beyond that the set is deliberately
    // generous — `+` in particular is both harmless and already used for
    // composite names like `rocjitsu+node+mi350x`.
    //
    // No separate `..` rule is needed. Escaping requires either a
    // separator — which the set above forbids — or a leading `..`, which
    // the leading-`.` rule above already rejects. What is left, `a..b`,
    // is an ordinary single component and harmless.
    if let Some(c) = name
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+')))
    {
        return bad(&format!("it contains {c:?}; allowed: [A-Za-z0-9._+-]"));
    }
    Ok(())
}

/// Reject a name mirage would have to rewrite before storing it.
///
/// Profiles and agents are addressed case-insensitively and live at a
/// lowercase path, so `MyProfile` and `myprofile` are one document. The
/// name inside the document therefore has to be the lowercase one, and
/// quietly substituting it meant `mirage profile create MyProfile`
/// reported `MyProfile` and every later command showed `myprofile`. Ask
/// for the name that will actually be stored instead of inventing it.
fn validate_stored_name(kind: DocKind, name: &str) -> Result<()> {
    let canonical = kind.canonical(name);
    if canonical == name {
        return Ok(());
    }
    let shown_name = shown(name);
    let shown_canonical = shown(&canonical);
    let kind = kind.as_str();
    Err(MirageError::other(format!(
        "invalid {kind} name {shown_name}: {kind}s are addressed case-insensitively and \
         stored lowercase, so mirage would save this one as {shown_canonical} and every \
         later command would show you a name you did not type. Use {shown_canonical}."
    )))
}

/// Validate every document reference a profile carries.
///
/// A reference is a name someone else's path is built from — following
/// `agent: "../../outside/evil"` reads a file outside the config
/// directory just as surely as asking for that agent by name would. The
/// front door was guarded and this was not, so a profile could be created
/// with exit 0 and escape only later, when something resolved it.
///
/// # Errors
///
/// Returns a rejection naming the reference and what is wrong with it.
pub fn validate_profile_refs(profile: &ProfileDef) -> Result<()> {
    match &profile.emulator.topology {
        MaybeRef::Ref(name) => validate_name(DocKind::Topology, name),
        MaybeRef::Owned(topology) => validate_topology_refs(topology),
    }
}

/// Validate the agent reference a topology carries. See
/// [`validate_profile_refs`].
///
/// # Errors
///
/// Returns a rejection naming the reference and what is wrong with it.
pub fn validate_topology_refs(topology: &TopologyDef) -> Result<()> {
    match &topology.agent {
        MaybeRef::Ref(name) => validate_name(DocKind::Agent, name),
        MaybeRef::Owned(_) => Ok(()),
    }
}

/// The document a reference was read out of.
///
/// A dangling reference is a fact about *two* documents, and the one a
/// user recognises is the one they named: told only that "a topology
/// refers to the agent \"ghost\"", someone with a dozen topologies has to
/// grep the config directory to find out which. The name is optional
/// because the resolvers are also reached from paths that genuinely do
/// not know it — a profile arriving over a run's socket carries a
/// topology reference that no name of the caller's ever passed through.
#[derive(Debug, Clone, Copy)]
pub struct Referrer<'a> {
    /// The kind of document the reference was found in.
    kind: DocKind,
    /// Its name, when whoever followed the reference knew it.
    name: Option<&'a str>,
}

impl<'a> Referrer<'a> {
    /// The referring document, named.
    #[must_use]
    pub fn named(kind: DocKind, name: &'a str) -> Self {
        Self {
            kind,
            name: Some(name),
        }
    }

    /// A referring document of this kind whose name is not known here.
    #[must_use]
    pub fn anonymous(kind: DocKind) -> Self {
        Self { kind, name: None }
    }

    /// How an error names it: the definite article and the name when
    /// there is one, and the indefinite article when there is not.
    fn describe(self) -> String {
        match self.name {
            Some(name) => format!("the {} {}", self.kind.as_str(), shown(name)),
            None => format!("a {}", self.kind.as_str()),
        }
    }
}

/// Report a reference that points at a document which is not there.
///
/// A reference is followed by interpolating it into a path, so a missing
/// target surfaced as `io error on <path>: No such file or directory` —
/// the filesystem's account of an operation nobody asked for, naming a
/// file the user has never typed and cannot connect to anything they did
/// type. What actually happened is that one document names another that
/// does not exist, and none of the four things the reader needs — which
/// document did the referring, which name failed to resolve, where mirage
/// looked, and how to see the names that would have worked — is in the
/// errno.
///
/// The referring *kind* is a property of where the reference lives rather
/// than of the call: a topology is referred to by a profile, and an agent
/// by a topology. The referring *name* is not, which is why it travels in
/// with the call. See [`crate::topology::store::get_referred_by`] and
/// [`crate::agent::store::get_referred_by`].
#[must_use]
pub fn dangling_ref(referrer: Referrer<'_>, missing: DocKind, name: &str) -> MirageError {
    let referrer = referrer.describe();
    let kind = missing.as_str();
    MirageError::other(format!(
        "dangling {kind} reference: {referrer} refers to the {kind} {}, and there is \
         no such {kind} (mirage looked for {}). Run `mirage {kind} list` for the \
         {kind} names this machine has.",
        shown(name),
        missing.path(name).display()
    ))
}

/// List every profile name, sorted.
///
/// # Errors
///
/// Returns an error if the profile directory exists but cannot be read.
pub fn profile_list() -> Result<Vec<String>> {
    list_json_stems(&crate::paths::profile_root())
}

/// Read one profile.
///
/// # Errors
///
/// Returns [`MirageError::ProfileNotFound`] if there is no such profile,
/// or a parse error if it is malformed.
pub fn profile_get(name: &str) -> Result<ProfileDef> {
    validate_name(DocKind::Profile, name)?;
    let path = crate::paths::profile_path(name);
    if !path.exists() {
        return Err(MirageError::not_found(DocKind::Profile, name));
    }
    crate::state::read_json(&path)
}

/// Write a new profile.
///
/// Everything a profile can be wrong about is decided here, before it
/// lands: the name, the references inside it, whether its emulator
/// backend accepts it, and whether it would replace something the user
/// wrote. That covers `profile import` as well as `profile create` — an
/// imported profile used to skip the emulator check entirely and fail
/// much later, at run time.
///
/// # Errors
///
/// Returns an error if the name or a reference is invalid, if the
/// emulator rejects the profile, if a profile of that name already exists
/// and is not an untouched builtin, or if the document cannot be written.
pub fn profile_put(profile: &ProfileDef) -> Result<Stored> {
    validate_name(DocKind::Profile, &profile.name)?;
    validate_stored_name(DocKind::Profile, &profile.name)?;
    validate_profile_refs(profile)?;
    profile
        .validate()
        .map_err(|e| MirageError::other(format!("profile {}: {e}", shown(&profile.name))))?;
    put_document(DocKind::Profile, &profile.name, profile)
}

/// Delete a profile.
///
/// # Errors
///
/// Returns [`MirageError::ProfileNotFound`] if there is no such profile,
/// or a rejection if it is an untouched builtin (which mirage would
/// simply write back — see `guard_delete`).
pub fn profile_delete(name: &str) -> Result<()> {
    validate_name(DocKind::Profile, name)?;
    let path = crate::paths::profile_path(name);
    if !path.exists() {
        return Err(MirageError::not_found(DocKind::Profile, name));
    }
    guard_delete(DocKind::Profile, name)?;
    std::fs::remove_file(&path).map_err(|e| MirageError::io(path, e))
}

/// List every topology name, sorted.
///
/// # Errors
///
/// Returns an error if the topology directory cannot be read.
pub fn topology_list() -> Result<Vec<String>> {
    crate::topology::store::list()
}

/// Read one topology.
///
/// # Errors
///
/// Returns an error if there is no such topology, or it is malformed.
pub fn topology_get(name: &str) -> Result<TopologyDef> {
    validate_name(DocKind::Topology, name)?;
    let path = crate::paths::topology_path(name);
    if !path.exists() {
        return Err(MirageError::not_found(DocKind::Topology, name));
    }
    crate::topology::store::get(name)
}

/// Write a new topology under `name`.
///
/// Guarded exactly as [`profile_put`] is: the three resource verbs are
/// parallel, so a topology someone edited is refused rather than
/// discarded. This one used to be the odd one out and went straight to
/// disk.
///
/// The agent reference is *resolved* here as well, not merely checked for
/// shape. A profile naming a topology that does not exist is refused when
/// it is written — the emulator backend follows the chain to answer
/// "can you run this?" — and the same user writing a topology naming an
/// agent that does not exist got exit 0 and a document that fails at
/// every later command. Two parallel verbs cannot hold references to
/// different standards.
///
/// # Errors
///
/// Returns an error if the name or the agent reference is invalid, if the
/// agent reference resolves to nothing, if a topology of that name
/// already exists and is not an untouched builtin, or if the document
/// cannot be written.
pub fn topology_put(name: &str, topology: &TopologyDef) -> Result<Stored> {
    validate_name(DocKind::Topology, name)?;
    validate_topology_refs(topology)?;
    resolve_topology_refs(Referrer::named(DocKind::Topology, name), topology)?;
    put_document(DocKind::Topology, name, topology)
}

/// Refuse a topology whose agent reference points at nothing.
///
/// Existence, not a parse: this answers "is there an agent by that name",
/// and an agent that is on disk but malformed is a different complaint,
/// owed to whoever reads it rather than to whoever writes a topology
/// beside it. Reading it here would also make writing a topology depend
/// on every agent staying parseable by *this* build.
fn resolve_topology_refs(referrer: Referrer<'_>, topology: &TopologyDef) -> Result<()> {
    let MaybeRef::Ref(agent) = &topology.agent else {
        return Ok(());
    };
    if DocKind::Agent.path(agent).exists() {
        return Ok(());
    }
    Err(dangling_ref(referrer, DocKind::Agent, agent))
}

/// Delete a topology.
///
/// # Errors
///
/// Returns an error if there is no such topology, or if it is an
/// untouched builtin (which mirage would simply write back — see
/// `guard_delete`).
pub fn topology_delete(name: &str) -> Result<()> {
    validate_name(DocKind::Topology, name)?;
    let path = crate::paths::topology_path(name);
    if !path.exists() {
        return Err(MirageError::not_found(DocKind::Topology, name));
    }
    guard_delete(DocKind::Topology, name)?;
    // Collected before the file goes, so a delete that fails warns about
    // nothing; reported after, so the warning describes what happened
    // rather than what was about to.
    let referrers = referrers_to(DocKind::Topology, name);
    std::fs::remove_file(&path).map_err(|e| MirageError::io(path, e))?;
    warn_broken_referrers(DocKind::Topology, name, &referrers);
    Ok(())
}

/// List every agent name, sorted.
///
/// # Errors
///
/// Returns an error if the agent directory cannot be read.
pub fn agent_list() -> Result<Vec<String>> {
    crate::agent::store::list()
}

/// Read one agent.
///
/// # Errors
///
/// Returns an error if there is no such agent, or it is malformed.
pub fn agent_get(name: &str) -> Result<AgentDef> {
    validate_name(DocKind::Agent, name)?;
    let path = crate::paths::agent_path(name);
    if !path.exists() {
        return Err(MirageError::not_found(DocKind::Agent, name));
    }
    crate::agent::store::get(name)
}

/// Write a new agent under `name`.
///
/// # Errors
///
/// Returns an error if the name is invalid, if an agent of that name
/// already exists and is not an untouched builtin, or if the document
/// cannot be written.
pub fn agent_put(name: &str, agent: &AgentDef) -> Result<Stored> {
    validate_name(DocKind::Agent, name)?;
    validate_stored_name(DocKind::Agent, name)?;
    put_document(DocKind::Agent, name, agent)
}

/// Delete an agent.
///
/// # Errors
///
/// Returns an error if there is no such agent, or if it is an untouched
/// builtin (which mirage would simply write back — see `guard_delete`).
pub fn agent_delete(name: &str) -> Result<()> {
    validate_name(DocKind::Agent, name)?;
    let path = crate::paths::agent_path(name);
    if !path.exists() {
        return Err(MirageError::not_found(DocKind::Agent, name));
    }
    guard_delete(DocKind::Agent, name)?;
    // See [`topology_delete`] for why this is read before the removal and
    // said after it.
    let referrers = referrers_to(DocKind::Agent, name);
    std::fs::remove_file(&path).map_err(|e| MirageError::io(path, e))?;
    warn_broken_referrers(DocKind::Agent, name, &referrers);
    Ok(())
}

/// The `.json` stems in a directory, sorted.
///
/// A missing directory reads as empty rather than as an error: it just
/// means nothing of that kind has been written yet, which is the normal
/// state of a fresh machine.
fn list_json_stems(root: &Path) -> Result<Vec<String>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root).map_err(|e| MirageError::io(root, e))? {
        let entry = entry.map_err(|e| MirageError::io(root, e))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(stem) = name.strip_suffix(".json") {
            out.push(stem.to_string());
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::emulator::{EmulatorDef, ExecMode};

    /// The backend name core's own tests write profiles against; see
    /// `crate::emulator::tests`.
    const TEST_EMULATOR: &str = "test";

    fn profile(name: &str) -> ProfileDef {
        profile_referring_to(name, MaybeRef::Ref("t".to_string()))
    }

    fn profile_referring_to(name: &str, topology: MaybeRef<TopologyDef>) -> ProfileDef {
        ProfileDef {
            name: name.to_string(),
            description: None,
            emulator: EmulatorDef {
                emulator: TEST_EMULATOR.to_string(),
                plugins: Default::default(),
                exec_mode: ExecMode::Functional,
                options: Default::default(),
                topology,
            },
            containerize: None,
        }
    }

    /// A [`BuiltinDocuments`] provider standing in for `mirage_builtin`,
    /// which core cannot depend on. Registering it here is what lets these
    /// tests exercise the builtin rules at all: without a provider the
    /// registry is empty and every document on disk is the user's.
    const SEEDED_PROFILE: &str = "seeded";

    fn seeded_documents() -> Vec<(DocKind, String, serde_json::Value)> {
        vec![(
            DocKind::Profile,
            SEEDED_PROFILE.to_string(),
            serde_json::to_value(profile(SEEDED_PROFILE)).unwrap(),
        )]
    }

    inventory::submit! {
        BuiltinDocuments { documents: seeded_documents }
    }

    /// Write the shipped copy of [`SEEDED_PROFILE`] to disk the way
    /// `mirage_builtin` does on startup — around the store, so the store's
    /// own guards do not get a say.
    fn seed() {
        crate::state::write_json(
            &crate::paths::profile_path(SEEDED_PROFILE),
            &profile(SEEDED_PROFILE),
        )
        .unwrap();
    }

    #[test]
    fn profiles_round_trip() {
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        assert!(profile_list().unwrap().is_empty());
        assert_eq!(profile_put(&profile("a")).unwrap(), Stored::Created);
        profile_put(&profile("b")).unwrap();
        assert_eq!(profile_list().unwrap(), vec!["a", "b"]);
        assert_eq!(profile_get("a").unwrap().name, "a");

        profile_delete("a").unwrap();
        assert_eq!(profile_list().unwrap(), vec!["b"]);
        assert!(matches!(
            profile_get("a"),
            Err(MirageError::ProfileNotFound { .. })
        ));
        crate::paths::clear_test_root();
    }

    #[test]
    fn profile_names_are_case_insensitive_to_read() {
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        profile_put(&profile("mixedcase")).unwrap();
        assert_eq!(profile_list().unwrap(), vec!["mixedcase"]);
        assert_eq!(profile_get("MIXEDCASE").unwrap().name, "mixedcase");
        crate::paths::clear_test_root();
    }

    #[test]
    fn a_name_that_would_be_rewritten_is_refused_rather_than_rewritten() {
        // `mirage profile create MyProfile` used to report "created
        // profile MyProfile" and create `myprofile` — the name it echoed
        // back was not the name it stored, and no later command would
        // ever show the one the user typed.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        let err = profile_put(&profile("MyProfile")).unwrap_err().to_string();
        assert!(err.contains("\"MyProfile\""), "{err}");
        assert!(err.contains("\"myprofile\""), "{err}");
        assert!(profile_list().unwrap().is_empty(), "nothing may be stored");

        let err = agent_put("MI350X", &AgentDef::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("\"mi350x\""), "{err}");
        assert!(agent_list().unwrap().is_empty());

        // Topologies are stored verbatim, so their case is nobody's
        // business but the user's.
        seed_agent();
        topology_put("MI350X-1x8", &topology()).unwrap();
        assert_eq!(topology_list().unwrap(), vec!["MI350X-1x8"]);
        crate::paths::clear_test_root();
    }

    #[test]
    fn a_profile_is_never_overwritten_without_being_asked() {
        // These files are the only copy: a profile someone tuned and then
        // recreated under the same name used to vanish, with exit 0 and
        // no message.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        let mut tuned = profile("tuned");
        tuned.description = Some("six months of tuning".to_string());
        profile_put(&tuned).unwrap();

        let err = profile_put(&profile("tuned")).unwrap_err().to_string();
        assert!(err.contains("already exists"), "{err}");
        assert!(err.contains("mirage profile delete tuned"), "{err}");
        assert_eq!(profile_get("tuned").unwrap(), tuned, "the user's copy");

        // And once it is gone, the name is free again.
        profile_delete("tuned").unwrap();
        profile_put(&profile("tuned")).unwrap();
        crate::paths::clear_test_root();
    }

    #[test]
    fn a_topology_is_guarded_like_a_profile_and_an_agent() {
        // The three resource verbs are parallel, and a user has no way to
        // infer that one of them destroys their work where the other two
        // refuse. `topology create` was the odd one out: it went straight
        // to disk, so an edited builtin topology could be discarded by a
        // command that printed `created topology …` and exited 0.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());
        seed_agent();

        let mut mine = topology();
        mine.gpus_per_node = 3;
        topology_put("mine", &mine).unwrap();

        let err = topology_put("mine", &topology()).unwrap_err().to_string();
        assert!(err.contains("already exists"), "{err}");
        assert!(err.contains("mirage topology delete mine"), "{err}");
        assert_eq!(
            topology_get("mine").unwrap().gpus_per_node,
            3,
            "the user's copy survived"
        );

        topology_delete("mine").unwrap();
        topology_put("mine", &topology()).unwrap();
        crate::paths::clear_test_root();
    }

    #[test]
    fn a_builtin_is_a_seed_a_user_may_take_over() {
        // A builtin nobody has touched is mirage's own copy, so replacing
        // it destroys nothing — that is how a builtin gets customised.
        // Once it *is* the user's, it is protected like any other.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());
        seed();
        assert!(is_pristine_builtin(DocKind::Profile, SEEDED_PROFILE));

        let mut mine = profile(SEEDED_PROFILE);
        mine.description = Some("mine now".to_string());
        // Allowed, but never in silence: the caller has to be able to say
        // that a shipped document just left the disk.
        assert_eq!(profile_put(&mine).unwrap(), Stored::ReplacedBuiltin);
        assert!(!is_pristine_builtin(DocKind::Profile, SEEDED_PROFILE));

        let err = profile_put(&profile(SEEDED_PROFILE))
            .unwrap_err()
            .to_string();
        assert!(err.contains("already exists"), "{err}");
        crate::paths::clear_test_root();
    }

    #[test]
    fn deleting_an_untouched_builtin_is_refused_rather_than_faked() {
        // Mirage rewrites every missing builtin on the next command, so
        // this delete used to report success, remove the file, and have
        // the file back before the user could look.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());
        seed();

        let err = profile_delete(SEEDED_PROFILE).unwrap_err().to_string();
        assert!(err.contains("is a builtin"), "{err}");
        assert!(err.contains(SEEDED_PROFILE), "{err}");
        assert!(
            crate::paths::profile_path(SEEDED_PROFILE).exists(),
            "the refusal must leave the file alone"
        );
        // A rejection is not a 404: the name does exist.
        assert!(!profile_delete(SEEDED_PROFILE).unwrap_err().is_not_found());

        // Editing it makes the delete meaningful again — the user's
        // version really does go, and that is how a builtin is reset.
        let mut mine = profile(SEEDED_PROFILE);
        mine.description = Some("mine now".to_string());
        profile_put(&mine).unwrap();
        profile_delete(SEEDED_PROFILE).unwrap();
        assert!(!crate::paths::profile_path(SEEDED_PROFILE).exists());
        crate::paths::clear_test_root();
    }

    #[test]
    fn a_reference_cannot_escape_the_config_directory() {
        // `validate_name` guarded the name a user types. A reference
        // *inside* a document is interpolated into a path by the same
        // rule and was not guarded at all, so `mirage profile create trav
        // --agent ../../outside/evil` was accepted with exit 0 and
        // resolved outside the config root when something followed it.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        let escaping = TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Ref("../../outside/evil".to_string()),
        };
        let inline = profile_referring_to("trav", MaybeRef::Owned(escaping.clone()));
        let err = profile_put(&inline).unwrap_err().to_string();
        assert!(err.contains("../../outside/evil"), "{err}");
        assert!(profile_list().unwrap().is_empty());

        let by_name = profile_referring_to("trav", MaybeRef::Ref("../../etc/passwd".to_string()));
        assert!(profile_put(&by_name).is_err());
        assert!(topology_put("trav", &escaping).is_err());

        // And wherever a reference is *followed*, not just where one is
        // written: a document that reached the disk some other way is
        // still resolved through these.
        assert!(crate::agent::store::get("../../outside/evil").is_err());
        assert!(crate::topology::store::get("../../outside/evil").is_err());
        crate::paths::clear_test_root();
    }

    #[test]
    fn a_profile_no_emulator_can_run_is_refused_at_the_door() {
        // Both `profile create` and `profile import` land here, which is
        // the point: an imported profile used to skip this check entirely
        // and fail at `mirage run`, long after the file was accepted.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        let mut p = profile("bogus");
        p.emulator.emulator = "no-such-emulator".to_string();
        let err = profile_put(&p).unwrap_err().to_string();
        assert!(err.contains("no-such-emulator"), "{err}");
        assert!(err.contains("mirage emulators"), "{err}");
        assert!(profile_list().unwrap().is_empty());
        crate::paths::clear_test_root();
    }

    #[test]
    fn an_unknown_field_is_reported_rather_than_dropped() {
        // A typo'd key used to be discarded in silence, so the emulated
        // machine was quietly not the one the file described.
        let err = serde_json::from_str::<ProfileDef>(
            r#"{"name":"p","descriptoin":"typo",
                "emulator":{"emulator":"test","plugins":{},
                            "exec_mode":"Functional","options":{},
                            "topology":"t"}}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("descriptoin"), "{err}");

        let err = serde_json::from_str::<TopologyDef>(
            r#"{"num_nodes":2,"gpus_pernode":8,"agent":"MI350X"}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("gpus_pernode"), "{err}");

        let err = serde_json::from_str::<AgentDef>(
            r#"{"vm":{"arch":"cdna4","gpu":{"num_xdcs":8}},"topology":{"root":{"name":"soc","type":"soc"}}}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("num_xdcs"), "{err}");
    }

    fn topology() -> TopologyDef {
        TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Ref("MI350X".to_string()),
        }
    }

    /// Put the agent [`topology`] refers to on disk.
    ///
    /// `topology_put` resolves that reference rather than merely checking
    /// its shape, so a test that stores a topology has to store its agent
    /// first — which is the order a user works in too.
    fn seed_agent() {
        agent_put("mi350x", &AgentDef::default()).unwrap();
    }

    #[test]
    fn a_missing_directory_lists_as_empty() {
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());
        // A fresh machine has written nothing yet; that is not an error.
        assert!(profile_list().unwrap().is_empty());
        assert!(agent_list().unwrap().is_empty());
        crate::paths::clear_test_root();
    }

    #[test]
    fn a_name_cannot_escape_its_directory() {
        // These names arrive off the daemon's socket. Interpolated into
        // `<config>/profile/<name>.json`, `..` walks out of the config
        // directory and a leading `/` replaces it — so `profile_get`
        // would read, and `profile_delete` would delete, an arbitrary
        // file the user has access to.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        let escapes = [
            "../../../etc/passwd",
            "..",
            "/etc/passwd",
            "a/b",
            ".hidden",
            "",
        ];
        for name in escapes {
            assert!(profile_get(name).is_err(), "profile_get({name:?})");
            assert!(profile_delete(name).is_err(), "profile_delete({name:?})");
            assert!(topology_get(name).is_err(), "topology_get({name:?})");
            assert!(agent_delete(name).is_err(), "agent_delete({name:?})");
            // A rejection must not be reported as "not found": that reads
            // as a name the caller may safely create.
            assert!(
                !profile_get(name).unwrap_err().is_not_found(),
                "profile_get({name:?}) must reject, not 404"
            );
        }

        // Ordinary names still work.
        profile_put(&profile("mi350x.tuned_v2-a")).unwrap();
        assert_eq!(
            profile_get("mi350x.tuned_v2-a").unwrap().name,
            "mi350x.tuned_v2-a"
        );

        crate::paths::clear_test_root();
    }

    #[test]
    fn only_one_of_two_racing_creates_reports_success() {
        // The guard read the directory and the write filled it, and
        // another `mirage … create` of the same name fitted between the
        // two: eight workers produced two "successes", and the document
        // left on disk was the later writer's — so the process told it had
        // created the topology had not created the one that exists.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());
        seed_agent();

        const WORKERS: u32 = 8;
        let start = std::sync::Barrier::new(WORKERS as usize);
        let winners = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            let start = &start;
            let winners = &winners;
            for gpus in 1..=WORKERS {
                scope.spawn(move || {
                    let mut mine = topology();
                    mine.gpus_per_node = gpus;
                    start.wait();
                    if topology_put("contested", &mine).is_ok() {
                        winners.lock().unwrap().push(gpus);
                    }
                });
            }
        });

        let winners = winners.into_inner().unwrap();
        assert_eq!(winners.len(), 1, "exactly one create may report success");
        // And the winner is the document on disk, which is the half that
        // makes the report worth anything.
        assert_eq!(topology_get("contested").unwrap().gpus_per_node, winners[0]);
        crate::paths::clear_test_root();
    }

    #[test]
    fn an_enormous_name_is_cut_down_before_it_is_quoted_back() {
        // A shell expansion that produced a 5000-character argument used
        // to produce a 5000-character error line, which scrolls the
        // sentence explaining the problem out of the terminal.
        let huge = "x".repeat(5000);
        let err = validate_name(DocKind::Profile, &huge)
            .unwrap_err()
            .to_string();
        assert!(err.len() < 200, "{} characters", err.len());
        assert!(err.contains("5000 characters"), "{err}");
        assert!(err.contains("xxxx"), "the prefix identifies it: {err}");

        // A name short enough to read is still quoted whole.
        let err = validate_name(DocKind::Profile, "has a space")
            .unwrap_err()
            .to_string();
        assert!(err.contains("\"has a space\""), "{err}");
    }

    #[test]
    fn a_dangling_reference_is_reported_as_one_rather_than_as_an_errno() {
        // Following a reference interpolates it into a path, so a profile
        // whose topology is not on disk failed with `io error on
        // /…/topology/ghosttopo.json: No such file or directory` — the
        // filesystem's account of an operation the user never asked for,
        // naming a file they had never typed. None of what they need is in
        // that: which kind of document did the referring, which name did
        // not resolve, and how to see the names that would have worked.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        let err = crate::topology::store::get("ghosttopo")
            .unwrap_err()
            .full_message();
        assert!(!err.contains("io error"), "{err}");
        assert!(err.contains("dangling topology reference"), "{err}");
        assert!(err.contains("a profile refers to"), "{err}");
        assert!(err.contains("ghosttopo"), "{err}");
        assert!(err.contains("mirage topology list"), "{err}");

        // Anonymous because nothing one frame up knew a profile name:
        // the indefinite article is the honest form, and inventing a name
        // would be worse than not having one.
        let err = crate::agent::store::get("ghostagent")
            .unwrap_err()
            .full_message();
        assert!(!err.contains("io error"), "{err}");
        assert!(err.contains("dangling agent reference"), "{err}");
        assert!(err.contains("a topology refers to"), "{err}");
        assert!(err.contains("ghostagent"), "{err}");
        assert!(err.contains("mirage agent list"), "{err}");

        crate::paths::clear_test_root();
    }

    #[test]
    fn a_topology_naming_an_agent_that_is_not_there_is_refused_like_a_profile() {
        // The two write verbs held references to different standards.
        // `profile create` follows the chain (the emulator backend has to,
        // to answer "can you run this?") and refuses a reference that
        // resolves to nothing; `topology create` checked only that the
        // name was a legal filename and wrote the document, so the user
        // got exit 0 and a topology that fails at every later command.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        let mut ghostly = topology();
        ghostly.agent = MaybeRef::Ref("ghostagent".to_string());
        let err = topology_put("mine", &ghostly).unwrap_err().to_string();
        assert!(err.contains("dangling agent reference"), "{err}");
        assert!(err.contains("ghostagent"), "{err}");
        assert!(err.contains("mirage agent list"), "{err}");
        assert!(
            topology_list().unwrap().is_empty(),
            "a refused topology may not reach the disk"
        );

        // The same document is accepted once the agent it names exists,
        // which is what makes the refusal a diagnosis rather than a ban.
        agent_put("ghostagent", &AgentDef::default()).unwrap();
        topology_put("mine", &ghostly).unwrap();
        assert_eq!(topology_list().unwrap(), vec!["mine"]);

        // An inline agent refers to nothing and is nobody's to resolve.
        let mut inline = topology();
        inline.agent = MaybeRef::Owned(AgentDef::default());
        topology_put("inline", &inline).unwrap();
        crate::paths::clear_test_root();
    }

    #[test]
    fn a_dangling_reference_names_the_document_that_holds_it() {
        // "a topology refers to the agent \"ghostagent\"" is unactionable
        // on a machine with a dozen topologies: the reader has to grep the
        // config directory to find out which one, and they had just typed
        // its name on the command line.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        let mut ghostly = topology();
        ghostly.agent = MaybeRef::Ref("ghostagent".to_string());
        let err = topology_put("mi350x-1x8", &ghostly)
            .unwrap_err()
            .to_string();
        assert!(err.contains("the topology \"mi350x-1x8\""), "{err}");

        // And directly, for the callers that resolve a reference rather
        // than write one.
        let err = crate::agent::store::get_referred_by(
            Referrer::named(DocKind::Topology, "mi350x-1x8"),
            "ghostagent",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("the topology \"mi350x-1x8\""), "{err}");

        let err = crate::topology::store::get_referred_by(
            Referrer::named(DocKind::Profile, "cdna4"),
            "ghosttopo",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("the profile \"cdna4\""), "{err}");
        assert!(err.contains("ghosttopo"), "{err}");

        crate::paths::clear_test_root();
    }

    #[test]
    fn deleting_a_document_reports_what_still_refers_to_it() {
        // A delete used to succeed in silence and break every document
        // that named the victim, which the user then met as an error
        // about a reference, in an unrelated command, with nothing
        // connecting it to what they had done.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());
        seed_agent();

        topology_put("by-name", &topology()).unwrap();
        // Spelled the other way: agents are addressed case-insensitively,
        // so this is the same reference written differently and has to be
        // found as one.
        let mut lowercase = topology();
        lowercase.agent = MaybeRef::Ref("mi350x".to_string());
        topology_put("lowercase", &lowercase).unwrap();

        // A profile reaching the agent through an inline topology is the
        // same breakage to whoever runs it.
        let inline = profile_referring_to("inline", MaybeRef::Owned(topology()));
        profile_put(&inline).unwrap();
        // …and a profile naming the topology directly is a referrer of
        // *that*.
        profile_put(&profile_referring_to(
            "byname",
            MaybeRef::Ref("by-name".to_string()),
        ))
        .unwrap();

        assert_eq!(
            referrers_to(DocKind::Agent, "MI350X"),
            vec![
                Reference {
                    kind: DocKind::Profile,
                    name: "inline".to_string()
                },
                Reference {
                    kind: DocKind::Topology,
                    name: "by-name".to_string()
                },
                Reference {
                    kind: DocKind::Topology,
                    name: "lowercase".to_string()
                },
            ]
        );
        assert_eq!(
            referrers_to(DocKind::Topology, "by-name"),
            vec![Reference {
                kind: DocKind::Profile,
                name: "byname".to_string()
            }]
        );

        // Nothing refers to a profile, and a document nobody names is not
        // a warning waiting to happen.
        assert!(referrers_to(DocKind::Profile, "inline").is_empty());
        assert!(referrers_to(DocKind::Agent, "never-referenced").is_empty());

        // The delete still goes through: these files are the user's.
        agent_delete("mi350x").unwrap();
        assert!(agent_list().unwrap().is_empty());
        crate::paths::clear_test_root();
    }

    #[test]
    fn a_rejected_name_is_reported_under_the_kind_it_belongs_to() {
        // `validate_name` took the kind as a string, so the word in the
        // message was whatever the call site happened to type rather than
        // the one word `DocKind::as_str` defines for that kind. Nothing
        // stopped a resource verb from telling a user their *topology*
        // name was an invalid profile name.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        for (kind, err) in [
            (DocKind::Profile, profile_get("has a space").unwrap_err()),
            (DocKind::Topology, topology_get("has a space").unwrap_err()),
            (DocKind::Agent, agent_get("has a space").unwrap_err()),
            (
                DocKind::Topology,
                crate::topology::store::get("has a space").unwrap_err(),
            ),
            (
                DocKind::Agent,
                crate::agent::store::get("has a space").unwrap_err(),
            ),
        ] {
            let err = err.to_string();
            assert!(
                err.starts_with(&format!("invalid {} name", kind.as_str())),
                "{err}"
            );
        }

        crate::paths::clear_test_root();
    }

    #[test]
    fn every_kind_reports_a_missing_document_as_not_found() {
        // Not merely "an error": the kind is what the HTTP API turns into
        // a 404 rather than a 500, what survives the wire in
        // `proto::ErrorKind`, and what `is_not_found` answers for a
        // caller cleaning up something that may already be gone. Reading
        // or deleting a missing agent used to surface as a raw
        // `io error on /…/agent/ghost.json`, which is none of those.
        let _g = crate::paths::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        crate::paths::set_test_root(dir.path());

        for err in [
            profile_delete("ghost").unwrap_err(),
            topology_delete("ghost").unwrap_err(),
            agent_delete("ghost").unwrap_err(),
            profile_get("ghost").unwrap_err(),
            topology_get("ghost").unwrap_err(),
            agent_get("ghost").unwrap_err(),
        ] {
            assert!(err.is_not_found(), "{err:?}");
        }

        crate::paths::clear_test_root();
    }
}
