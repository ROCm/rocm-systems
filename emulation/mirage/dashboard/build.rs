use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

const GENERATED_ASSETS_RS: &str = "dashboard_assets.rs";
const SPA_OUT_DIR: &str = "dashboard-spa";

struct Asset {
    path: String,
    full_path: PathBuf,
    content_type: &'static str,
}

fn main() {
    // The SPA is only built and embedded when the `webui` feature is
    // enabled. Without it this crate is an empty library, so we skip
    // the Node.js toolchain entirely and let `cargo build` stay fast
    // and dependency-free.
    if env::var_os("CARGO_FEATURE_WEBUI").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let web_dir = manifest_dir.join("web");
    let spa_out_dir = out_dir.join(SPA_OUT_DIR);

    emit_rerun_directives(&web_dir);
    check_node_version();
    install_web_dependencies_if_needed(&web_dir);
    build_spa(&web_dir, &spa_out_dir);

    let assets = collect_assets(&spa_out_dir);
    if assets.is_empty() {
        panic!(
            "dashboard SPA build produced no files in {}",
            spa_out_dir.display()
        );
    }

    write_assets_module(&out_dir.join(GENERATED_ASSETS_RS), &assets)
        .unwrap_or_else(|err| panic!("failed to write embedded dashboard assets: {err}"));
}

fn emit_rerun_directives(web_dir: &Path) {
    println!("cargo:rerun-if-env-changed=NPM");
    println!("cargo:rerun-if-env-changed=NODE");
    println!("cargo:rerun-if-env-changed=MIRAGE_DASHBOARD_SKIP_NPM_CI");

    for path in [
        web_dir.join("index.html"),
        web_dir.join("package.json"),
        web_dir.join("package-lock.json"),
        web_dir.join("tsconfig.json"),
        web_dir.join("tsconfig.app.json"),
        web_dir.join("tsconfig.node.json"),
        web_dir.join("vite.config.ts"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    emit_rerun_for_tree(&web_dir.join("public"));
    emit_rerun_for_tree(&web_dir.join("src"));
}

fn emit_rerun_for_tree(path: &Path) {
    if !path.exists() {
        return;
    }

    let entries =
        fs::read_dir(path).unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));

    for entry in entries {
        let entry =
            entry.unwrap_or_else(|err| panic!("failed to read entry in {}: {err}", path.display()));
        let entry_path = entry.path();
        let file_type = entry
            .file_type()
            .unwrap_or_else(|err| panic!("failed to inspect {}: {err}", entry_path.display()));

        if file_type.is_dir() {
            emit_rerun_for_tree(&entry_path);
        } else if file_type.is_file() {
            println!("cargo:rerun-if-changed={}", entry_path.display());
        }
    }
}

/// Minimum Node.js major.minor required to build the SPA. Vite 8 and
/// the React 19 toolchain require Node 20.19+ (or 22.12+); building on
/// an older runtime fails with cryptic errors, so we check up front
/// and emit an actionable message instead.
const MIN_NODE_MAJOR: u32 = 20;
const MIN_NODE_MINOR: u32 = 19;

fn check_node_version() {
    // Allow CI/sandboxes that pre-build the SPA to skip the toolchain
    // entirely (same escape hatch used for `npm ci`).
    if env::var_os("MIRAGE_DASHBOARD_SKIP_NPM_CI").is_some() {
        return;
    }

    let node = env::var_os("NODE").unwrap_or_else(|| OsString::from("node"));
    let output = match Command::new(&node).arg("--version").output() {
        Ok(output) => output,
        Err(err) => panic!(
            "could not run `{} --version` to verify the Node.js toolchain: {err}.\n\
             Building the mirage dashboard requires Node.js {MIN_NODE_MAJOR}.{MIN_NODE_MINOR}+ \
             and npm on PATH. Install Node.js (see emulation/mirage/docs/building.md), or set \
             MIRAGE_DASHBOARD_SKIP_NPM_CI=1 if the SPA is prebuilt.",
            node.to_string_lossy(),
        ),
    };

    if !output.status.success() {
        panic!(
            "`node --version` failed with status {}; is Node.js installed and on PATH?",
            output.status
        );
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let version = raw.trim().trim_start_matches('v');
    let (major, minor) = parse_node_version(version).unwrap_or_else(|| {
        panic!("could not parse Node.js version from `node --version` output: {raw:?}")
    });

    if (major, minor) < (MIN_NODE_MAJOR, MIN_NODE_MINOR) {
        panic!(
            "Node.js {major}.{minor} is too old to build the mirage dashboard; \
             {MIN_NODE_MAJOR}.{MIN_NODE_MINOR}+ is required (Vite 8 / React 19 toolchain).\n\
             Upgrade Node.js (see emulation/mirage/docs/building.md), or set \
             MIRAGE_DASHBOARD_SKIP_NPM_CI=1 if the SPA is prebuilt."
        );
    }
}

/// Parse a `major.minor[.patch]` version string into `(major, minor)`.
fn parse_node_version(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

fn install_web_dependencies_if_needed(web_dir: &Path) {
    if env::var_os("MIRAGE_DASHBOARD_SKIP_NPM_CI").is_some() {
        return;
    }

    let package_lock = web_dir.join("package-lock.json");
    let installed_lock = web_dir.join("node_modules").join(".package-lock.json");
    let install_needed = match (modified(&package_lock), modified(&installed_lock)) {
        (Some(package_lock_time), Some(installed_lock_time)) => {
            package_lock_time > installed_lock_time
        }
        (Some(_), None) => true,
        _ => false,
    };

    if !install_needed {
        return;
    }

    let mut command = Command::new(npm());
    command.arg("ci").current_dir(web_dir);
    run_command(command, "npm ci");
}

fn modified(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok()?.modified().ok()
}

fn build_spa(web_dir: &Path, spa_out_dir: &Path) {
    if spa_out_dir.exists() {
        fs::remove_dir_all(spa_out_dir)
            .unwrap_or_else(|err| panic!("failed to clear {}: {err}", spa_out_dir.display()));
    }

    let mut command = Command::new(npm());
    command
        .arg("run")
        .arg("build")
        .arg("--")
        .arg("--outDir")
        .arg(spa_out_dir)
        .arg("--emptyOutDir")
        .current_dir(web_dir);

    run_command(command, "npm run build");
}

fn npm() -> OsString {
    env::var_os("NPM").unwrap_or_else(|| OsString::from("npm"))
}

fn run_command(mut command: Command, label: &str) {
    let status = command
        .status()
        .unwrap_or_else(|err| panic!("failed to run {label}: {err}"));

    if !status.success() {
        panic!("{label} failed with status {status}");
    }
}

fn collect_assets(root: &Path) -> Vec<Asset> {
    let mut assets = Vec::new();
    collect_assets_from(root, root, &mut assets);
    assets.sort_by(|left, right| left.path.cmp(&right.path));
    assets
}

fn collect_assets_from(root: &Path, path: &Path, assets: &mut Vec<Asset>) {
    let entries =
        fs::read_dir(path).unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));

    for entry in entries {
        let entry =
            entry.unwrap_or_else(|err| panic!("failed to read entry in {}: {err}", path.display()));
        let entry_path = entry.path();
        let file_type = entry
            .file_type()
            .unwrap_or_else(|err| panic!("failed to inspect {}: {err}", entry_path.display()));

        if file_type.is_dir() {
            collect_assets_from(root, &entry_path, assets);
        } else if file_type.is_file() {
            let path = asset_path(root, &entry_path);
            let content_type = content_type_for(&path);
            assets.push(Asset {
                path,
                full_path: entry_path,
                content_type,
            });
        }
    }
}

fn asset_path(root: &Path, path: &Path) -> String {
    let relative_path = path
        .strip_prefix(root)
        .unwrap_or_else(|err| panic!("failed to relativize {}: {err}", path.display()));
    let mut parts = Vec::new();

    for component in relative_path.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_str().unwrap_or_else(|| {
                panic!("dashboard asset path is not UTF-8: {}", path.display())
            })),
            _ => panic!(
                "unexpected dashboard asset path component in {}",
                path.display()
            ),
        }
    }

    parts.join("/")
}

fn content_type_for(path: &str) -> &'static str {
    let extension = Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("");

    match extension {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" | "cjs" => "text/javascript; charset=utf-8",
        "json" | "map" => "application/json; charset=utf-8",
        "txt" => "text/plain; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "wasm" => "application/wasm",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "otf" => "font/otf",
        _ => "application/octet-stream",
    }
}

fn write_assets_module(path: &Path, assets: &[Asset]) -> io::Result<()> {
    let mut file = fs::File::create(path)?;

    writeln!(file, "#[derive(Debug, Clone, Copy, PartialEq, Eq)]")?;
    writeln!(file, "pub struct DashboardAsset {{")?;
    writeln!(file, "    pub path: &'static str,")?;
    writeln!(file, "    pub content_type: &'static str,")?;
    writeln!(file, "    pub bytes: &'static [u8],")?;
    writeln!(file, "}}")?;
    writeln!(file)?;
    writeln!(file, "pub static ASSETS: &[DashboardAsset] = &[")?;

    for asset in assets {
        writeln!(file, "    DashboardAsset {{")?;
        writeln!(file, "        path: {:?},", asset.path)?;
        writeln!(file, "        content_type: {:?},", asset.content_type)?;
        writeln!(
            file,
            "        bytes: include_bytes!({:?}),",
            asset.full_path.display().to_string()
        )?;
        writeln!(file, "    }},")?;
    }

    writeln!(file, "];")?;
    writeln!(file)?;
    writeln!(
        file,
        "pub fn get(path: &str) -> Option<&'static DashboardAsset> {{"
    )?;
    writeln!(file, "    let path = path.trim_start_matches('/');")?;
    writeln!(
        file,
        "    let path = if path.is_empty() {{ \"index.html\" }} else {{ path }};"
    )?;
    writeln!(file, "    ASSETS.iter().find(|asset| asset.path == path)")?;
    writeln!(file, "}}")?;
    writeln!(file)?;
    writeln!(
        file,
        "pub fn get_spa(path: &str) -> Option<&'static DashboardAsset> {{"
    )?;
    writeln!(file, "    get(path).or_else(|| get(\"index.html\"))")?;
    writeln!(file, "}}")?;

    Ok(())
}
