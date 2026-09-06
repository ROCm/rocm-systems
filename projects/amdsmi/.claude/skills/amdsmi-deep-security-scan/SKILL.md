Perform a deep, read-only security audit of a SINGLE ROCm component directory.
The component has already been partitioned for you by the calling script. Your job is full-coverage analysis of THIS directory tree only — not org-wide discovery, not triage, not deciding what to skip.

---

## Goal

Find security bugs, vulnerabilities, insecure patterns, dependency risks, secrets, memory-safety issues, trust-boundary violations, and product-security weaknesses in the given ROCm component. Produce a single consolidated Markdown report.

---

## Scope

* Scan root: the repository path supplied below.
* Recurse into EVERY subdirectory, including nested git submodules.
* Do NOT skip any directory for being "vendored", "upstream fork", "generated", or "lower priority".
* If a subtree is too large for deep semantic analysis, fall back to pattern/grep-based scanning and report it as `Partial (pattern-only)` in the Coverage Map — never silently omit it.
* Prioritize C/C++, GPU runtime, compiler/toolchain, drivers, and privileged system-interaction code.
* CI/CD, IaC, build scripts, containers, and packaging are secondary but must still be covered.

---

## Safety and Execution Constraints

This is a **READ-ONLY** security analysis task.

**Allowed:**
* Read files, traverse directories, inspect git metadata/history
* Run `grep`/`rg`/`find` and similar read-only inspection
* Static analysis only
* Generate Markdown/JSON/CSV reports

**Do NOT:**
* Modify, commit, push, or delete any files inside the scan root
* Install packages or execute untrusted binaries
* Run build systems unless strictly required for static analysis
* Run repository scripts, containers, or network operations
* Perform active exploitation or fuzzing
* Open pull requests

The ONLY permitted write is the report file at `REPORT_FILE` (path supplied by the calling script, outside the scan root).
If command execution is necessary, use minimum privilege, never modify repo contents, never run unknown scripts.

---

## Execution Strategy

You are scanning ONE component. Do not spend turns on repo discovery or cross-repo partitioning — go straight to analysis.

1. **Inventory** — enumerate every top-level directory under the scan root with file counts and dominant language. This becomes the Coverage Map skeleton.
2. **Mandatory pattern sweep** — run ALL grep patterns below across the FULL tree. Record every hit location.
3. **Targeted deep semantic analysis** — for each directory, in priority order, read and reason about security-relevant source.
4. **Hardening / build-config audit** — every `CMakeLists.txt` / `*.cmake`.
5. **GitHub Actions deep audit** — every `.github/workflows/*.yml` file with full semantic analysis.
6. **Secret detection sweep** — entropy scan + pattern sweep across all file types including binary-adjacent configs.
7. **Container security audit** — every `Dockerfile*`, `docker-compose*.yml`, `.dockerignore`.
8. **IaC security audit** — every `*.tf`, `*.tfvars`, `*.tfvars.json`, Helm charts, Kustomize, Ansible playbooks, Pulumi/CDK configs.
9. **Vendored dependency version extraction** — for `third-party/`, `external/`, bundled libs, and all manifest files.
10. **Repository governance file audit** — check for the presence of `.gitignore`, `CODEOWNERS`, `LICENSE` (any casing), `CONTRIBUTING.md`, and `SECURITY.md` at the repository root. Record `Present` or `Missing` for each. Missing files are Low-severity findings reported in the Security Engineering Gaps section.
11. **Introduced-by annotation** — for every finding confirmed as non-benign, run `git log -S '<key pattern>' --follow --oneline -- <file>` and `git blame -L <line>,<line> <file>` to identify the introducing commit SHA, author, and date. Record these in the `**Introduced:**` field of each finding. This step is read-only and must not modify the repository.
12. **Report assembly** — fill the output template, including the Coverage Map with `Full` / `Partial` / `None` for every directory.

Increase depth dynamically when you encounter: dangerous APIs, attacker-reachable parsers, privileged paths, memory-management hot spots, IPC/shared-memory, ELF/loader code, env-var handling, or dlopen/plugin loading.

---

## Mandatory Pattern Sweep

This sweep is a **minimum baseline**, not the full scan. It guarantees every known-dangerous API and secret pattern is located. It does NOT replace deep semantic analysis — you must still apply your full security knowledge beyond these patterns.

Run all patterns via `grep -rn` / `rg -n` over the entire tree before semantic analysis. Report every hit location even if later judged benign — mark confidence per finding.

### C/C++ Memory-Safety Patterns

```
\b(strcpy|strcat|sprintf|vsprintf|gets|scanf|strtok)\b
\b(memcpy|memmove|memset)\b\s*\([^,]+,[^,]+,[^)]*\b(len|size|n|count)\b
\b(system|popen|execl|execlp|execv|execvp|fork)\s*\(
\bdlopen\s*\(|\bdlsym\s*\(|\bLoadLibrary\w*\s*\(
\bgetenv\s*\(|\bsecure_getenv\s*\(
\bmmap\s*\([^)]*PROT_EXEC
\balloca\s*\(
reinterpret_cast<[^>]*\*>
(uint32_t|uint16_t|int|short)\s*\)\s*\w*(size|len|length|offset|count)
\b(chmod|fchmod)\s*\([^)]*0?7[0-7]{2}|\bsetuid\b|\bseteuid\b|\bsetgid\b
\b(tmpnam|tempnam|mktemp)\b
\b(read|write|pread|pwrite|recv|send)\s*\([^)]*,\s*\w+\s*,\s*\w+\s*\)
ioctl\s*\(
TODO|FIXME|HACK|XXX
```

### GitHub Actions Patterns (run against `.github/workflows/**` only)

```
uses:\s*[^@]+@(master|main|HEAD|v?\d+)\s*$
pull_request_target
workflow_run
\$\{\{\s*github\.event\.(issue\.body|pull_request\.body|head_commit\.message|head_commit\.author|comment\.body|review\.body|pages\[)
\$\{\{\s*github\.event\.pull_request\.head\.ref
\$\{\{\s*github\.event\.pull_request\.head\.sha
run:\s*.*\$\{\{[^}]+\}\}
GITHUB_TOKEN.*write
permissions:\s*write-all
permissions:\s*contents:\s*write
secrets\s*:\s*inherit
ACTIONS_ALLOW_UNSECURE_COMMANDS\s*:\s*true
actions/upload-artifact(?!.*sha256)
actions/download-artifact(?!.*sha256)
cache(?!.*key.*\$\{\{\s*hashFiles)
self-hosted
runs-on:\s*\[.*self-hosted
environment:(?!.*protection_rules)
```

### Secret Detection Patterns (run across ALL files including configs, scripts, YAML, JSON, TOML, .env)

```
# Generic credential keywords
(?i)(password|passwd|pwd|secret|api[_-]?key|api[_-]?secret|auth[_-]?token|access[_-]?token|private[_-]?key|client[_-]?secret)\s*[:=]\s*['"]?[A-Za-z0-9+/=_\-]{8,}

# AWS
AKIA[0-9A-Z]{16}
(?i)aws[_-]?(secret|access)[_-]?key\s*[:=]\s*['"]?[A-Za-z0-9/+=]{40}
(?i)aws[_-]?session[_-]?token\s*[:=]

# GCP
-----BEGIN (RSA |EC )?PRIVATE KEY-----
"type"\s*:\s*"service_account"
AIza[0-9A-Za-z\-_]{35}

# Azure
(?i)AccountKey=[A-Za-z0-9+/=]{88}==
DefaultEndpointsProtocol=https;AccountName=
(?i)(azure|az)[_-]?(client[_-]?secret|tenant[_-]?id|subscription[_-]?id)\s*[:=]\s*['"]?[0-9a-f\-]{32,}

# GitHub / GitLab
ghp_[A-Za-z0-9]{36}
ghs_[A-Za-z0-9]{36}
github_pat_[A-Za-z0-9_]{82}
glpat-[A-Za-z0-9_\-]{20}

# PyPI / NPM
pypi-AgEI[A-Za-z0-9_\-]{60,}
npm_[A-Za-z0-9]{36}

# High-entropy strings (flag for manual review)
['"][A-Za-z0-9+/]{40,}['"]
['"][A-Za-z0-9_\-]{32,}['"]

# PEM blocks and SSH keys
-----BEGIN [A-Z ]*PRIVATE KEY
-----BEGIN OPENSSH PRIVATE KEY
-----BEGIN CERTIFICATE

# Insecure secret storage locations
\.env$
\.netrc$
\.aws/credentials
\.kube/config
id_rsa$|id_ecdsa$|id_ed25519$

# Secrets in Docker ENV/ARG
^\s*(ENV|ARG)\s+\w*(SECRET|PASSWORD|KEY|TOKEN|PASS)\w*\s*=

# Secrets in CI/CD env blocks (outside secrets: context)
env:\s*\n\s+\w*(SECRET|PASSWORD|KEY|TOKEN)\w*\s*:

curl .*\|\s*(ba)?sh|wget .*\|\s*(ba)?sh
```

### Container Security Patterns (run against `Dockerfile*`, `docker-compose*.yml`, `*.yaml`)

```
FROM\s+\S+:(latest|master|main|stable|edge)\s
^FROM\s+(?!.*@sha256:)
\bADD\s+https?://
^RUN.*curl.*\|\s*(ba)?sh
^RUN.*wget.*\|\s*(ba)?sh
^RUN.*apt-get install(?!.*--no-install-recommends)
^\s*USER\s+root
(?m)^(?!.*USER ).*CMD|(?m)^(?!.*USER ).*ENTRYPOINT
privileged:\s*true
--privileged
cap_add:
SYS_ADMIN|SYS_PTRACE|NET_ADMIN|ALL
--cap-add
network_mode:\s*host
pid:\s*host
volumes:.*(/var/run/docker\.sock|/proc|/sys|/dev)
security_opt:.*no-new-privileges:\s*false
seccomp:\s*unconfined
apparmor:\s*unconfined
read_only:\s*false
```

### IaC / Terraform / Kubernetes Patterns

```
# Terraform — public exposure
acl\s*=\s*"public-read"
block_public_acls\s*=\s*false
restrict_public_buckets\s*=\s*false
publicly_accessible\s*=\s*true

# Terraform — overly permissive IAM
"Action"\s*:\s*"\*"
"Resource"\s*:\s*"\*"
"Effect"\s*:\s*"Allow"

# Terraform — missing encryption
encrypted\s*=\s*false
storage_encrypted\s*=\s*false
enable_disk_encryption\s*=\s*false
sse_algorithm\s*=\s*"aws:kms"

# Terraform — network exposure
cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]
ipv6_cidr_blocks\s*=\s*\["::/0"\]

# Terraform — hardcoded credentials
(?i)(access_key|secret_key|password|token)\s*=\s*"[^$][^"]{8,}"

# Kubernetes — privileged / host access
privileged:\s*true
hostPID:\s*true
hostIPC:\s*true
hostNetwork:\s*true
allowPrivilegeEscalation:\s*true
runAsNonRoot:\s*false
readOnlyRootFilesystem:\s*false
automountServiceAccountToken:\s*true

# Kubernetes — overly permissive RBAC
verbs:\s*\[.*"\*"
resources:\s*\[.*"\*"
apiGroups:\s*\[.*"\*"

# Kubernetes — missing resource limits
(?m)^(?!.*limits).*containers:

# Helm — insecure defaults
\.Values\.\w*(password|secret|key|token)\b
```

### General Injection / Shell Patterns

```
\beval\s*\(
\bexec\s*\(
\bos\.system\s*\(
\bsubprocess\.(call|run|Popen)\s*\(.*shell\s*=\s*True
\bpickle\.(load|loads)\s*\(
\byaml\.load\s*\([^,)]+\)(?!\s*,\s*Loader=yaml\.SafeLoader)
\bjson\.loads\s*\(.*user
__import__\s*\(
\bXMLParser\b|\bxml\.etree|\bxml\.dom
```

---

## High-Priority Scan Targets

### Source code

* `*.c`, `*.cc`, `*.cpp`, `*.cxx`, `*.h`, `*.hh`, `*.hpp`, `*.hxx`, `*.inc`, `*.def`
* `*.cu`, `*.hip` — HIP/CUDA kernels
* `*.cl`, `*.ocl` — OpenCL kernels
* `*.S`, `*.s`, `*.asm` — hand-written assembly
* `*.td` — LLVM TableGen
* `*.ll`, `*.mlir` — IR (codegen injection surface)
* `*.py`, `*.sh` — automation invoking subprocess / os.system / eval
* `*.rb`, `*.pl`, `*.lua` — scripting languages in build/tooling context

### Build / CI / IaC

* `CMakeLists.txt`, `*.cmake`
* `.github/workflows/**/*.yml` and `.github/workflows/**/*.yaml`
* `.github/actions/**` — composite actions
* `Dockerfile*`, `docker-compose*.yml`, `.dockerignore`
* `*.tf`, `*.tfvars`, `*.tfvars.json`, `*.tfstate` (flag if committed)
* `chart/**`, `helm/**`, `charts/**` — Helm chart templates and values
* `k8s/**`, `kubernetes/**`, `manifests/**`, `kustomization.yaml`
* `ansible/**`, `playbooks/**`, `*.playbook.yml`
* `pulumi/**`, `Pulumi.yaml`, `*.ts` in IaC context
* `*.yaml`, `*.yml`, `*.json`, `*.toml` — config files
* `requirements*.txt`, `pyproject.toml`, `package*.json`, `package-lock.json`, `yarn.lock`
* `Cargo.toml`, `Cargo.lock`, `go.mod`, `go.sum`
* `Pipfile`, `Pipfile.lock`, `poetry.lock`, `conda*.yml`, `environment.yml`
* Release / signing / packaging scripts

---

## GitHub Actions Deep Audit

For every `.github/workflows/*.yml` and `.github/actions/**` file, perform full semantic analysis covering:

### Dangerous Triggers
* `pull_request_target` — grants write access to secrets for code from forks; any `run:` step that checks out fork code makes it RCE on the runner. Flag every occurrence and check if the workflow checks out `${{ github.event.pull_request.head.sha }}`.
* `workflow_run` — same risk profile as `pull_request_target`; the triggered workflow runs with the base repo's permissions. Verify the triggering workflow name and whether it checks out untrusted code.
* `schedule` + elevated permissions — check if scheduled workflows have broader permissions than needed.

### Script Injection via Expression Contexts
Audit every `run:` block for unquoted `${{ }}` expressions interpolating user-controlled data:
* `github.event.issue.body`, `github.event.pull_request.body`, `github.event.head_commit.message`, `github.event.comment.body`, `github.event.review.body`
* `github.event.pull_request.head.ref` (branch name is attacker-controlled)
* `github.event.pull_request.head.sha` (used in checkout — ensure this is the *merge* SHA, not the head SHA, for `pull_request_target`)

Any of these appearing in a `run:` shell command without using an intermediate env variable is a script injection vulnerability.

### Unpinned Actions
* Every `uses:` referencing a mutable ref (`@master`, `@main`, `@v1`, `@HEAD`) is a supply-chain risk. Actions MUST be pinned to a full commit SHA (`@abc1234`).
* Composite actions inside `.github/actions/` should also be verified for unpinned third-party `uses:` references.

### GITHUB_TOKEN and Permissions
* Workflows lacking a top-level `permissions:` block inherit `write-all` by default in many repo configs — flag this.
* `permissions: write-all` is overly broad — flag for principle of least privilege.
* `contents: write` or `packages: write` scopes should be justified.
* `secrets: inherit` in reusable workflows passes ALL caller secrets to the callee — flag every occurrence.

### Self-Hosted Runner Risks
* `runs-on: self-hosted` or `runs-on: [self-hosted, ...]` — flag all occurrences.
* Verify: Are these ephemeral (fresh VM per run) or persistent? Persistent runners with fork PR access are high risk.
* Check for credential mounts or host volume mounts in runner job containers.
* Check for `ACTIONS_ALLOW_UNSECURE_COMMANDS: true` — re-enables deprecated `set-env`/`add-path` which are command injection vectors.

### Artifact Integrity
* `actions/upload-artifact` followed by `actions/download-artifact` without SHA256 verification of the artifact content — flag as potential artifact poisoning.
* Cache actions without a deterministic key (`hashFiles(...)`) — flag for cache poisoning risk.

### PAT vs. App Authentication
* Hardcoded PATs in `env:` blocks (outside `secrets:` context) — flag as credential exposure.
* PATs used where OIDC (`id-token: write` + cloud provider OIDC) or GitHub Apps would be appropriate.

---

## Secret Detection Deep Audit

Perform a layered secret detection sweep beyond the mandatory pattern grep:

### Entropy Analysis
For each file type below, flag any string literal or value meeting both criteria:
* Length ≥ 20 characters
* Shannon entropy ≥ 4.5 bits/char (approximated: characters are not uniformly distributed among [a-z], [A-Z], [0-9], and symbols)

Apply entropy analysis to: `*.yml`, `*.yaml`, `*.json`, `*.toml`, `*.env`, `*.cfg`, `*.ini`, `*.conf`, `.github/workflows/**`, `Dockerfile*`, `*.tf`, `*.tfvars`, `*.sh`, `*.py`.

### High-Risk Secret Locations
Check these locations explicitly:
* `.env`, `.env.*`, `.env.local`, `.env.production` — these should never be committed
* `.netrc` — stores plaintext credentials for network hosts
* `~/.aws/credentials` patterns in any committed file
* `.kube/config` — kubeconfig with embedded certificates/tokens
* Private key files: `id_rsa`, `id_ecdsa`, `id_ed25519`, `*.pem`, `*.key`, `*.p12`, `*.pfx`
* `*.tfstate`, `*.tfstate.backup` — Terraform state often contains plaintext secrets
* `git log --all --full-history -- '*secret*' '*password*' '*credential*'` — check git history for deleted secret files

### Cloud Provider Token Patterns
Verify each provider's specific format:
* **AWS**: `AKIA...` (access key), 40-char base64 (secret key), session tokens
* **GCP**: `AIza...` (API key), service account JSON (`"type": "service_account"`), oauth2 tokens
* **Azure**: connection strings (`AccountKey=`), client secrets, SAS tokens (`sig=` in URLs)
* **GitHub**: `ghp_`, `ghs_`, `github_pat_` prefixes
* **PyPI**: `pypi-AgEI` prefix
* **NPM**: `npm_` prefix
* **Slack**: `xoxb-`, `xoxp-`, `xoxa-` prefixes
* **Stripe**: `sk_live_`, `pk_live_` prefixes
* **Twilio**: `SK` followed by 32 hex chars
* **Generic JWT**: three base64url segments separated by dots — flag if the header decodes to `alg: none` or `alg: HS256` with a weak/hardcoded key

### Secrets in Unexpected Locations
* Docker `ENV` and `ARG` instructions containing credential-like names — these are baked into the image layer and visible via `docker history`
* Build scripts that `echo` or `printf` credential values (they appear in build logs)
* CMake `message()` calls printing sensitive variables
* Debug/test fixtures with real-looking credentials
* Comments containing "use this token: ..." or similar

---

## SAST — Static Application Security Testing

### C/C++ Memory-Safety Analysis

* Buffer overflow / underflow, OOB read/write
* Use-after-free, double-free, dangling pointer
* Uninitialized memory read
* Null-pointer / wild-pointer dereference
* Integer overflow / underflow / truncation / sign-extension — pay special attention to narrowing casts applied to sizes, lengths, offsets, and indices
* Unsafe casts (`reinterpret_cast`, C-style, narrowing)
* Lifetime / ownership bugs, unsafe smart-pointer use, unsafe lambda captures
* Stack / heap corruption, alignment & strict-aliasing violations
* Format-string bugs (`printf(user_input)`, uncontrolled format specifiers)
* Unbounded allocation, `alloca()` / VLA on attacker-controlled size
* TOCTOU, race conditions, lock-order inversion, thread-unsafe shared state
* Atomic / memory-ordering correctness
* Missing bounds checks on array indexing and pointer arithmetic
* Unsafe deserialization / parsing of untrusted input
* Recursion without depth limit on untrusted input

### Taint-Flow Analysis (Interprocedural)

For each attacker-controlled data source (see ROCm-specific trust boundaries below, plus env vars, file input, network input, IPC), trace the data flow through function call chains to identify sinks:

**Sources** (attacker-controlled data entry points):
* `getenv()` / `secure_getenv()` return values
* `ioctl()` argument buffers copied from userspace
* AQL packet fields read from user-writable shared memory
* Code-object / ELF file bytes passed to `hipModuleLoad*` / `amd_comgr_*`
* `argv[]`, `envp[]` in any executable
* Data read from `recv()` / `read()` / file I/O where the file is user-supplied
* Plugin `.so` paths from env vars or config files

**Sinks** (dangerous operations):
* `memcpy` / `strcpy` / `sprintf` with tainted length or destination size
* Array indexing: `buf[tainted_index]`
* Pointer arithmetic: `ptr + tainted_offset`
* `system()` / `popen()` / `exec*()` with tainted argument
* `dlopen(tainted_path)`
* `mmap(tainted_length)`
* Format string: `printf(tainted_string)`
* SQL: string concatenation into query
* Any allocation: `malloc(tainted_size)` without overflow check

Document the full call chain from source to sink in each finding, even if it crosses multiple translation units. Mark as `Partial` if the chain cannot be fully resolved statically.

### Python / Shell Script Analysis

* `eval()`, `exec()` with any non-literal argument
* `os.system()`, `subprocess.call/run/Popen` with `shell=True` and any interpolated variable
* `pickle.load()` / `pickle.loads()` on untrusted data (arbitrary code execution)
* `yaml.load()` without `Loader=yaml.SafeLoader` (arbitrary code execution via `!!python/object`)
* `xml.etree.ElementTree`, `xml.dom`, `lxml` without disabling external entities (XXE)
* `__import__()` with a user-controlled module name
* Unvalidated `subprocess` commands built from user/env input
* `shlex.split()` missing where shell=False but arguments contain spaces
* Path traversal: `open(user_input)` without `os.path.abspath` + allowlist check
* Insecure deserialization via `marshal`, `shelve`, `jsonpickle`

### GPU / Kernel-Level Issues

* Shared memory (LDS) race conditions within a warp — accesses to shared memory without `__syncthreads()` barriers between write and read phases
* Warp-level information leakage — computations dependent on secret data that follow a data-dependent branch (warp divergence can leak timing)
* Missing `__threadfence()` / `__threadfence_system()` before host-visible flag writes
* Insufficient validation of kernel launch parameters (grid/block dimensions) before dispatch — integer overflow in `dim3` calculations
* Host↔device synchronization races: host reading device output before `hipDeviceSynchronize()` / stream sync
* Cross-process GPU memory leakage via unsanitized IPC handles (`hipIpcOpenMemHandle`)
* Device-side memory corruption reachable from host via AQL packets or doorbell writes

---

## Container Security Deep Audit

For every `Dockerfile*`, `docker-compose*.yml`, and `*.dockerignore` in the scan root:

### Base Image Analysis
* **Mutable tags**: `FROM image:latest`, `FROM image:stable` — flag; pin to `FROM image@sha256:<digest>`.
* **Unverified digest**: `FROM image:tag` without `@sha256:` — flag for supply-chain risk.
* **Root base images**: images known to run as root by default without explicit `USER` override — flag.
* **Bloated base images**: full OS images (`ubuntu`, `debian`, `centos`) where `distroless`, `alpine`, or `scratch` would suffice — flag as increased attack surface.
* **ROCm base images**: `rocm/dev-ubuntu-*`, `rocm/rocm-terminal` — flag the pinned version and note it should be verified against AMD's published digest.

### Dockerfile Instruction Audit
* `ADD <URL>` — fetches remote content at build time without integrity verification; use `COPY` + a verified download step instead.
* `ADD <archive>` — auto-extraction can be exploited with crafted archives; prefer explicit `COPY` + `tar`.
* `RUN curl ... | bash` or `RUN wget ... | sh` — arbitrary code execution at build time; flag all occurrences.
* `RUN apt-get install` without `--no-install-recommends` — installs unnecessary packages; flag.
* Missing `USER` instruction — container runs as root if not overridden; flag every `Dockerfile` where the last `USER` is `root` or absent.
* `COPY . .` without a `.dockerignore` — risk of including `.git`, `.env`, SSH keys, or other sensitive files in the image.
* `ENV SECRET=...` / `ARG API_KEY=...` — secrets baked into layers and visible via `docker history`; flag every credential-like `ENV`/`ARG`.
* Multi-stage builds: verify secrets used in a build stage are NOT copied to the final stage via `COPY --from=builder`.
* Missing `HEALTHCHECK` in long-running service containers.

### docker-compose Analysis
* `privileged: true` — grants full host capabilities; flag every occurrence.
* `network_mode: host` — bypasses network namespace isolation; flag.
* `pid: host` — shares host PID namespace; flag.
* Volume mounts exposing: `/var/run/docker.sock` (container escape), `/proc`, `/sys`, `/dev`, host home directories.
* `security_opt: [no-new-privileges: false]` or missing `no-new-privileges: true`.
* Missing `seccomp` profile (defaults to Docker's default seccomp, but should be explicit).
* Missing `read_only: true` for containers that don't need filesystem writes.
* `cap_add: [SYS_ADMIN]`, `cap_add: [ALL]`, `cap_add: [NET_ADMIN]` — flag all capability additions.
* `user: root` or missing `user:` directive.
* Services with `ports:` binding to `0.0.0.0` unnecessarily — flag for least-exposure.

### .dockerignore Audit
* Missing `.dockerignore` entirely — flag as high risk if `COPY . .` is used.
* `.dockerignore` present but not excluding: `.git`, `.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa`, `*.tfstate`, `node_modules`, `__pycache__`.

---

## IaC Security Deep Audit

### Terraform

For every `*.tf` and `*.tfvars` / `*.tfvars.json` file:

**State File Risks:**
* `*.tfstate` or `*.tfstate.backup` committed to the repository — flag as critical (state files contain plaintext secrets, resource IDs, and sometimes credentials).
* Backend configured with local state (`backend "local"`) — flag; remote encrypted state (S3 + KMS, GCS, Terraform Cloud) should be used.

**S3 / Storage Exposure:**
* `acl = "public-read"` or `acl = "public-read-write"` on any S3 bucket.
* `block_public_acls = false`, `block_public_policy = false`, `restrict_public_buckets = false`, `ignore_public_acls = false`.
* Missing server-side encryption: `server_side_encryption_configuration` block absent on S3 buckets.
* Missing versioning on S3 buckets storing release artifacts or build cache.

**IAM Privilege Escalation:**
* Policy documents with `"Action": "*"` and `"Effect": "Allow"`.
* Policy documents with `"Resource": "*"` on sensitive actions (IAM, KMS, S3 object delete, EC2 instance termination).
* Inline policies vs. managed policies — inline policies bypass permission boundaries.
* `iam:CreatePolicyVersion`, `iam:SetDefaultPolicyVersion`, `iam:AttachUserPolicy`, `iam:PassRole` granted broadly — these are privilege escalation vectors.
* Missing `condition` blocks on cross-account assume-role policies.

**Network Exposure:**
* Security groups with `cidr_blocks = ["0.0.0.0/0"]` on non-HTTP/HTTPS ports.
* RDS / ElastiCache / Redshift with `publicly_accessible = true`.
* Default VPC usage — flag; custom VPCs with explicit subnet configuration are required.

**Encryption at Rest / In Transit:**
* `encrypted = false` on EBS volumes, RDS instances, ElastiCache clusters.
* `storage_encrypted = false` on RDS.
* ALB/NLB listeners on port 80 without redirect to 443.
* `ssl_policy` absent or set to an outdated policy (anything older than `ELBSecurityPolicy-TLS13-1-2-2021-06`).

**Hardcoded Credentials:**
* Any string literal assigned to `access_key`, `secret_key`, `password`, `token`, `api_key` in `.tf` files — use variable references (`var.`) + secrets manager instead.

**Lifecycle / Deletion Protection:**
* Critical resources (databases, KMS keys, S3 buckets) missing `lifecycle { prevent_destroy = true }`.

### Kubernetes / Helm / Kustomize

**Pod Security:**
* Containers without `securityContext.runAsNonRoot: true`.
* Containers with `securityContext.privileged: true`.
* Containers with `securityContext.allowPrivilegeEscalation: true` (or not explicitly set to `false`).
* Containers with `securityContext.readOnlyRootFilesystem: false` (or not set).
* Pods with `hostPID: true`, `hostIPC: true`, `hostNetwork: true`.
* Containers without resource `limits` (CPU and memory) — denial of service risk.
* `automountServiceAccountToken: true` on pods that don't need API access.
* `hostPath` volume mounts — flag all; especially `/var/run/docker.sock`, `/etc`, `/proc`.

**RBAC:**
* `ClusterRole` or `Role` with `verbs: ["*"]` on sensitive resources — flag.
* `ClusterRole` with `resources: ["*"]` — flag.
* `ClusterRoleBinding` binding to `system:masters` group — equivalent to cluster admin.
* `ServiceAccount` bound to overly broad roles.
* Default `ServiceAccount` used (not a dedicated minimal-permission account).

**Network Policy:**
* Namespaces without a default-deny `NetworkPolicy` — all pods can communicate with all other pods by default.
* Ingress rules without TLS termination.

**Secrets Management:**
* Kubernetes `Secret` objects in YAML committed to the repository — flag; use Sealed Secrets, ESO (External Secrets Operator), or Vault.
* `kubectl create secret` commands hardcoded in scripts with literal values.

**Helm Charts:**
* `values.yaml` containing default passwords, tokens, or API keys — flag.
* Chart templates rendering `{{ .Values.password }}` directly into environment variables without a secret reference.
* Helm hooks with elevated RBAC permissions.
* Charts pinned to mutable `repository` tags without digest pinning.

**Kustomize:**
* `secretGenerator` with `literals:` — check if literal values are committed or templated from environment.

**Verify (R5):**
```bash
# Terraform — confirm no public S3 buckets after applying fix
terraform plan | grep -E "block_public|restrict_public|publicly_accessible"

# Confirm no committed state files
git ls-files | grep -E "\.tfstate(\.backup)?$"

# Kubernetes — confirm securityContext is set on all containers
kubectl get pods -A -o json | jq '
  .items[] | select(
    .spec.containers[].securityContext.runAsNonRoot != true
  ) | .metadata.name'

# Run checkov IaC scan to verify remaining issues
docker run --rm -v "$(pwd):/tf" bridgecrew/checkov:latest \
  -d /tf --framework terraform,kubernetes
```

**Prevent Regression (R5):**
```yaml
# .github/workflows/iac-scan.yml
- name: Checkov IaC scan
  uses: bridgecrewio/checkov-action@<SHA>
  with:
    directory: .
    framework: terraform,kubernetes,helm
    soft_fail: false

# Also add tfsec as a pre-commit hook for local dev:
# .pre-commit-config.yaml:
- repo: https://github.com/antonbabenko/pre-commit-terraform
  rev: <SHA>
  hooks:
    - id: terraform_tfsec
```

### Ansible

* `no_log: false` (or absent) on tasks handling passwords, tokens, or private keys — logs will expose credentials.
* `shell:` / `command:` modules with user-supplied variables — shell injection.
* `get_url:` without `checksum:` — integrity bypass.
* `become: yes` with `become_user: root` without justification — privilege escalation.
* Hardcoded credentials in `vars:`, `defaults/`, or `group_vars/` — should use Ansible Vault.
* `unsafe_writes: yes` on sensitive files.

---

## ROCm-Specific Trust Boundaries

Treat data on the untrusted side of each boundary as attacker-controlled:

* **KFD ioctl interface** — args copied from userspace into `libhsakmt` / kernel
* **AQL packet parsing & dispatch** — queue packets live in user-writable shared memory
* **Code-object / ELF loader** — `.hsaco` / `.co` / fatbin parsing (`hipModuleLoad*`, `hsa_code_object_*`, `amd_comgr_*`)
* **comgr** — compiles user-supplied source/bitcode at runtime
* **HIP/HSA env vars** — `HIP_*`, `HSA_*`, `ROCR_*`, `ROCM_*`, `LD_LIBRARY_PATH`, `LD_PRELOAD`, `HIP_VISIBLE_DEVICES`
* **rocm-smi / amdsmi** — privileged sysfs reads/writes, setuid helpers, device control
* **Doorbell / signal / event pages** mmap'd into userspace

---

## GPU / Runtime Security Focus

* GPU memory boundary & isolation validation
* Queue / command-submission validation, doorbell abuse
* Host↔device synchronization races
* Cross-process / cross-tenant GPU memory leakage
* Kernel-dispatch argument validation
* Device-side memory corruption reachable from host
* DMA / memory-mapping risks, `PROT_EXEC` mappings
* Driver↔runtime trust-boundary violations
* Unsafe firmware / microcode interaction
* GPU virtualization / partitioning (SR-IOV, MIG-equivalent) weaknesses
* Privileged compute / runtime escalation paths

---

## Compiler / Toolchain Focus

* Unsafe default compiler / linker flags
* Insecure build-time codegen, macro / template expansion injection
* Untrusted artifact ingestion (bitcode, object files, plugins)
* Unsafe caching / build reuse (poisoned cache → RCE)
* Insecure binary packaging / signing
* Trust-boundary violations in build system
* TableGen / `.td` / `.inc` generated code lacking bounds checks

---

## Build Hardening Audit

For every `CMakeLists.txt` / `*.cmake`, verify presence (or justify absence) of:

```
-fstack-protector-strong
-D_FORTIFY_SOURCE=2 (or =3)
-fPIE / -pie
-Wl,-z,relro -Wl,-z,now
-Wl,-z,noexecstack
-fstack-clash-protection
-fcf-protection=full
-Wformat -Wformat-security -Werror=format-security
-fvisibility=hidden (for libraries)
```

Flag any use of:

```
-fno-stack-protector
-z execstack
-Wno-format-security / -Wno-format
-U_FORTIFY_SOURCE / -D_FORTIFY_SOURCE=0
-O0 in release configs
--allow-shlib-undefined
```

---

## SCA — Software Composition Analysis

### Vendored / Bundled C++ Dependencies

Third-party code ships inside the ROCm product and is in scope for full source scanning — do NOT skip or downgrade it.

For each bundled dependency under `third-party/`, `external/`, `vendor/`, `third-party-sources/`, or fetched at build time from `rocm-third-party-deps.s3.us-east-2.amazonaws.com`:

* Extract name + pinned version (from `CMakeLists.txt`, `VERSION`, `git submodule status`, tarball name, or URL)
* Report as: `<lib> <version>` — note if >2 years stale
* List known CVEs to verify against (do not assert a CVE applies without evidence — list as "verify CVE-XXXX-YYYY")
* Run the full mandatory pattern sweep AND semantic analysis over extracted third-party source, same as first-party code
* Pay particular attention to: parsers (expat, msgpack, yaml-cpp, nlohmann-json, flatbuffers, sqlite3, elfio, elfutils), compression (zlib, zstd, bzip2, liblzma/xz), and anything linked into privileged components (libdrm, libpciaccess, libcap, numactl)
* Flag any local patches applied on top of upstream and review those diffs for security impact
* Verify each `URL` in CMake has a matching `URL_HASH` / checksum; flag any download without integrity verification

### Manifest-Based Dependencies

For each package manifest found (`requirements*.txt`, `pyproject.toml`, `package*.json`, `Cargo.toml`, `go.mod`, `Pipfile`, `conda*.yml`, `environment.yml`):

* Extract all direct dependencies with their pinned versions
* Extract all lock-file-resolved transitive dependencies where lock files exist (`package-lock.json`, `yarn.lock`, `Cargo.lock`, `go.sum`, `Pipfile.lock`, `poetry.lock`)
* Flag any dependency without a pinned version (e.g., `requests>=2.0` without an upper bound or lock file)
* Flag any dependency pinned to a commit SHA on a mutable branch (e.g., GitHub dependency on `main`)
* Flag dependencies with known CVEs (cross-reference against public advisories for each ecosystem)
* Flag packages that have been yanked or deprecated in their registry
* Flag >2-year-old pinned versions of security-sensitive packages (cryptography, requests, urllib3, paramiko, PyYAML, Pillow, lxml, and similar)

### Dependency Confusion / Typosquatting
* Identify any package names that could be confused with internal/private packages (short names, names that shadow stdlib modules, names differing by one character from well-known packages)
* Check for private packages fetched from public registries (pip install of an internal package name that could be hijacked on PyPI)
* Verify `--index-url` / `--extra-index-url` usage in pip configurations; flag if public PyPI is used as a fallback for private packages without `--no-index` isolation

### SBOM Gap
* Note if no SBOM (CycloneDX or SPDX format) is generated or committed for this component
* Note if no `SECURITY.md` or vulnerability disclosure policy exists

### Integrity Verification
* Flag any `FetchContent_Declare` / `ExternalProject_Add` in CMake without `URL_HASH` or `GIT_TAG` pinned to a commit SHA (mutable tags like `v1.2` can be moved)
* Flag any `pip install` in CI/CD scripts without `--require-hashes` or a lock file
* Flag any `go get` without pinned version in `go.mod`

---

## Security Engineering Gaps

* Components lacking fuzz harnesses (libFuzzer / AFL / OSS-Fuzz)
* Components lacking sanitizer build configs (ASan / UBSan / MSan / TSan)
* Parsers / loaders with no negative / malformed-input tests
* Missing hardening flags (per audit above)
* High-risk components lacking threat model / security docs
* Missing SBOM for any component
* No reproducible build configuration
* Artifact signing absent (Sigstore/cosign, GPG) for published binaries
* No SLSA provenance attestation for release artifacts
* No dependabot / Renovate configuration for automated dependency updates

---

## Repository Governance File Audit

Check for the following files at the repository root using exact-match search (case-insensitive for `LICENSE`). Record `Present` or `Missing` for each. A missing file is a **Low-severity** security engineering gap finding and must appear in Section 17 of the report.

```bash
# Run these checks against the scan root
for f in .gitignore CODEOWNERS CONTRIBUTING.md SECURITY.md; do
  [ -f "$SCAN_ROOT/$f" ] && echo "PRESENT: $f" || echo "MISSING: $f"
done
# LICENSE — accept any casing or extension
find "$SCAN_ROOT" -maxdepth 1 -iname "license*" -o -iname "copying*" | head -1 \
  && echo "PRESENT: LICENSE" || echo "MISSING: LICENSE"
```

| File | Security Purpose | Status | Verification Notes |
|---|---|---|---|
| `.gitignore` | Prevents accidental commit of secrets, credentials, build artifacts, and IDE files. Without it, broad `git add .` or `COPY . .` in Dockerfiles can leak sensitive material. | Present / Missing | If present, verify it excludes: `.env`, `*.pem`, `*.key`, `id_rsa`, `*.tfstate`, `__pycache__`, `node_modules`. Flag any absent exclusion as a sub-finding. |
| `CODEOWNERS` | Enforces mandatory reviewer assignment for security-sensitive paths; without it, PRs touching critical code can be merged without expert review. | Present / Missing | If present, verify that security-critical paths are covered: `src/`, `.github/workflows/`, `CMakeLists.txt`, `Dockerfile*`, `*.tf`. Flag uncovered paths. |
| `LICENSE` | Declares the legal distribution terms. Absence creates legal ambiguity for downstream consumers and open-source compliance programs. Accept: `LICENSE`, `LICENSE.md`, `LICENSE.txt`, `COPYING`. | Present / Missing | Note the license type (e.g., MIT, Apache-2.0, custom) — licenses that restrict security research or reverse engineering are a product-security concern. |
| `CONTRIBUTING.md` | Communicates secure development expectations to contributors: coding standards, how to handle security-sensitive changes, and the PR review process. | Present / Missing | If present, check whether it references the vulnerability disclosure process or links to `SECURITY.md`. |
| `SECURITY.md` | Defines the vulnerability disclosure policy (VDP). Its absence means security researchers have no sanctioned, documented path to report bugs — increasing the risk of public zero-day disclosure. GitHub surfaces this file in the Security tab. | Present / Missing | If present, verify it includes: a contact method (email or HackerOne/BugBounty URL), a response SLA, and a scope statement. Flag if any of these are absent. |

---

## General Security Focus

* Hardcoded secrets, accidental credential exposure, secrets in logs
* Shell / command injection, unsafe `subprocess` / `os.system` / `eval`
* Path traversal, symlink attacks, insecure temp files
* Insecure downloads (no checksum / TLS verification)
* Insecure defaults, missing input validation, missing authn/authz
* Supply-chain: artifact publishing / signing, provenance, SLSA gaps

---

## Risk Prioritization

Rank highest anything that could:

* Enable RCE on host or in CI
* Cause memory corruption reachable from untrusted input
* Leak GPU or host memory across tenants / processes
* Escape a trust boundary (user→kernel, guest→host, sandbox→host)
* Compromise build / release infrastructure (supply-chain)
* Expose credentials
* Allow privilege escalation
* Corrupt workloads or models in production HPC/AI environments

---

## Remediation Guidance

Every finding's **Remediation**, **Verification**, and **Prevent Regression** fields MUST include concrete examples drawn from the patterns below. Generic advice without a code/config snippet is not acceptable. Use these as templates — adapt file paths, variable names, and values to the actual finding.

### Unified diff format requirement

Code remediations MUST be formatted as unified diffs using the real file path from the scan root, so developers can apply them directly with `patch -p1`. Example structure:

```diff
--- a/src/loader/hip_module_loader.cpp
+++ b/src/loader/hip_module_loader.cpp
@@ -247,7 +247,8 @@ static int load_module(const char *path, char *dst, size_t dst_len) {
-  strcpy(dst, path);
+  if (strnlen(path, dst_len) >= dst_len) return -EINVAL;
+  strncpy(dst, path, dst_len - 1);
+  dst[dst_len - 1] = '\0';
```

For YAML, CMake, and other config files where a diff is awkward, a clearly labelled before/after block is acceptable instead.

---

### R1 — C/C++ Memory-Safety Remediations

**Unsafe string functions → bounded alternatives**
```c
// BEFORE (vulnerable)
strcpy(dst, src);
strcat(dst, src);
sprintf(buf, fmt, arg);

// AFTER (safe)
strncpy(dst, src, sizeof(dst) - 1);
dst[sizeof(dst) - 1] = '\0';
strncat(dst, src, sizeof(dst) - strlen(dst) - 1);
snprintf(buf, sizeof(buf), fmt, arg);
// Prefer: strlcpy / strlcat (BSD) or std::string in C++
```

**Integer overflow before allocation / indexing**
```c
// BEFORE (vulnerable — overflow → under-allocation)
char *buf = malloc(user_count * sizeof(Record));
buf[user_index] = ...;

// AFTER (safe)
if (user_count > SIZE_MAX / sizeof(Record))
    return -EINVAL;
if (user_index >= user_count)
    return -ERANGE;
char *buf = malloc(user_count * sizeof(Record));
```

**Use-after-free / double-free**
```c
// AFTER (safe pattern)
free(ptr);
ptr = NULL;   // prevents double-free and dangling-pointer use
// In C++: use std::unique_ptr<T> — prevents both automatically
```

**Format-string injection**
```c
// BEFORE (vulnerable)
printf(user_input);
syslog(LOG_INFO, user_message);

// AFTER (safe)
printf("%s", user_input);
syslog(LOG_INFO, "%s", user_message);
```

**Unbounded alloca / VLA on attacker-controlled size**
```c
// BEFORE (stack corruption)
void process(size_t n) { char buf[n]; ... }

// AFTER (safe)
#define MAX_BUF 4096
void process(size_t n) {
    if (n > MAX_BUF) return -EINVAL;
    char buf[MAX_BUF];  // fixed, or use malloc with bounds check
}
```

**TOCTOU on temporary files**
```c
// BEFORE (vulnerable)
char *path = tmpnam(NULL);
int fd = open(path, O_CREAT | O_WRONLY, 0600);

// AFTER (safe)
char template[] = "/tmp/rocm-XXXXXX";
int fd = mkstemp(template);  // atomic create + open, no race window
```

**Unsafe reinterpret_cast on user-controlled data**
```cpp
// BEFORE (aliasing / alignment UB)
MyStruct *s = reinterpret_cast<MyStruct *>(user_buf);

// AFTER (safe copy-based deserialization)
MyStruct s;
static_assert(sizeof(s) <= BUF_SIZE);
memcpy(&s, user_buf, sizeof(s));
// Then validate each field in s individually
```

**Verify (R1):**
```bash
# Confirm unsafe functions are gone from the affected file
grep -n "strcpy\|strcat\|sprintf\|gets" path/to/fixed_file.c

# Confirm stack protector is linked into the binary
readelf -s ./build/lib/librocm.so | grep __stack_chk_fail

# Run AddressSanitizer build to catch remaining memory bugs
cmake -DCMAKE_C_FLAGS="-fsanitize=address,undefined" -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc) && ./tests/unit_tests
```

**Prevent Regression (R1):**
```yaml
# .github/workflows/sast.yml — add Semgrep step
- name: Semgrep memory safety
  uses: semgrep/semgrep-action@<SHA>
  with:
    config: >-
      r/c.lang.security.insecure-use-strcpy
      r/c.lang.security.insecure-use-sprintf
      r/c.lang.security.insecure-use-gets

# CMakeLists.txt — enforce ASan in CI sanitizer build target
add_compile_options($<$<CONFIG:Sanitize>:-fsanitize=address,undefined>)
add_link_options($<$<CONFIG:Sanitize>:-fsanitize=address,undefined>)
```

---

### R2 — GitHub Actions Remediations

**Unpinned action → SHA-pinned**
```yaml
# BEFORE (mutable tag — supply-chain risk)
uses: actions/checkout@v4

# AFTER (immutable SHA pin)
uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
# Maintain with: https://github.com/mheap/pin-github-action or Dependabot
```

**Script injection via user-controlled expression**
```yaml
# BEFORE (RCE if issue body contains shell metacharacters)
- run: echo "${{ github.event.issue.body }}"

# AFTER (pass through env var — shell does NOT interpret the value)
- env:
    ISSUE_BODY: ${{ github.event.issue.body }}
  run: echo "$ISSUE_BODY"
```

**pull_request_target checking out fork code**
```yaml
# BEFORE (RCE: runs attacker's code with write permissions)
on: pull_request_target
jobs:
  build:
    steps:
      - uses: actions/checkout@...
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - run: ./build.sh   # attacker-controlled script

# AFTER (check out base repo only, or use separate read-only job)
on: pull_request_target
jobs:
  build:
    steps:
      - uses: actions/checkout@...  # checks out BASE repo, not fork
      - run: ./build.sh             # trusted code only
# If fork code is needed, do it in a separate job with NO secrets access
```

**Missing permissions block (defaults to write-all)**
```yaml
# AFTER — add at workflow top level and narrow per job
permissions:
  contents: read       # minimum for checkout
  # Add only what this workflow actually needs, e.g.:
  # pull-requests: write  # only if the job posts PR comments
  # id-token: write       # only if the job uses OIDC
```

**ACTIONS_ALLOW_UNSECURE_COMMANDS**
```yaml
# REMOVE this entirely — it re-enables deprecated set-env / add-path
# ACTIONS_ALLOW_UNSECURE_COMMANDS: true   ← DELETE

# Use the modern equivalents instead:
# echo "MY_VAR=value" >> $GITHUB_ENV
# echo "/my/path"     >> $GITHUB_PATH
```

**secrets: inherit in reusable workflow**
```yaml
# BEFORE (passes ALL caller secrets to callee)
jobs:
  call:
    uses: ./.github/workflows/reusable.yml
    secrets: inherit

# AFTER (pass only what the callee needs)
jobs:
  call:
    uses: ./.github/workflows/reusable.yml
    secrets:
      NEEDED_SECRET: ${{ secrets.NEEDED_SECRET }}
```

**Self-hosted persistent runner receiving fork PRs**
```yaml
# AFTER — gate fork PRs behind an approval environment
on: pull_request_target
jobs:
  test:
    environment: fork-pr-approval   # requires human approval before run
    runs-on: self-hosted
```

**Verify (R2):**
```bash
# Confirm no unpinned actions remain in workflows
grep -rn "uses:.*@\(master\|main\|HEAD\|v[0-9]\+\)\s*$" .github/workflows/

# Confirm no pull_request_target with checkout of fork SHA
grep -rn "pull_request_target" .github/workflows/

# Confirm permissions block exists in every workflow file
for f in .github/workflows/*.yml; do
  grep -q "^permissions:" "$f" || echo "MISSING permissions: in $f"
done

# Simulate injection: ensure ${{ github.event.issue.body }} is never
# used directly in a run: block (must go through env var)
grep -rn 'run:.*\${{.*github\.event\.' .github/workflows/
```

**Prevent Regression (R2):**
```yaml
# Add to .github/workflows/ci.yml
- name: Audit GitHub Actions security
  uses: ossf/scorecard-action@<SHA>
  with:
    results_file: scorecard.sarif
    publish_results: false

# Add zizmor to pre-commit for local workflow linting
# .pre-commit-config.yaml:
- repo: https://github.com/zizmorcore/zizmor
  rev: <SHA>
  hooks:
    - id: zizmor
```

---

### R3 — Secret Exposure Remediations

**Hardcoded credential in source / config**
```bash
# BEFORE
API_KEY="sk_live_abc123..."

# AFTER — reference a secrets manager at runtime
API_KEY="$(aws secretsmanager get-secret-value \
           --secret-id myapp/api_key --query SecretString --output text)"
# Or in CI: reference ${{ secrets.API_KEY }} and never print it
```

**Secret in Docker ENV / ARG (visible in docker history)**
```dockerfile
# BEFORE (baked into image layer)
ENV DATABASE_PASSWORD=supersecret

# AFTER — use BuildKit secrets (never stored in layer)
# RUN --mount=type=secret,id=db_pass \
#     DB_PASS=$(cat /run/secrets/db_pass) && ./configure ...
# Build with: docker build --secret id=db_pass,src=./secret.txt .
```

**Secret in CI env block outside secrets context**
```yaml
# BEFORE (printed in logs if step fails)
env:
  API_TOKEN: "ghp_abc123..."

# AFTER
env:
  API_TOKEN: ${{ secrets.API_TOKEN }}
# Ensure the secret is masked: GitHub auto-masks values registered as secrets
```

**.env file committed to repository**
```bash
# Immediate remediation steps:
# 1. Rotate the exposed credential immediately — assume it is compromised.
# 2. Remove the file: git rm --cached .env
# 3. Add to .gitignore:  echo ".env" >> .gitignore
# 4. Rewrite git history if the file was pushed: git filter-repo --path .env --invert-paths
# 5. Add a pre-commit hook or gitleaks CI scan to prevent recurrence.
```

**Private key committed**
```bash
# 1. Revoke/rotate the key immediately — it must be treated as compromised.
# 2. git rm --cached path/to/id_rsa
# 3. Rewrite history: git filter-repo --path path/to/id_rsa --invert-paths
# 4. Force-push all branches; require all contributors to re-clone.
# 5. Add *.pem, id_rsa, id_ecdsa, id_ed25519 to .gitignore.
```

**Verify (R3):**
```bash
# Confirm the secret no longer appears anywhere in git history
git log --all -S 'the_leaked_token_value' --oneline

# Confirm no high-entropy strings remain in config/YAML files
docker run --rm -v "$(pwd):/repo" trufflesecurity/trufflehog:latest \
  filesystem /repo --only-verified

# Confirm .env is in .gitignore and not tracked
git ls-files --error-unmatch .env 2>&1 | grep "not in the index"
```

**Prevent Regression (R3):**
```yaml
# .pre-commit-config.yaml — block secret commits locally
- repo: https://github.com/gitleaks/gitleaks
  rev: <SHA>
  hooks:
    - id: gitleaks

# .github/workflows/secrets-scan.yml — scan on every push
- name: Gitleaks secret scan
  uses: gitleaks/gitleaks-action@<SHA>
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

### R4 — Container Security Remediations

**Mutable base image tag → digest-pinned**
```dockerfile
# BEFORE (tag can be moved; supply-chain risk)
FROM rocm/dev-ubuntu-22.04:latest

# AFTER (immutable digest)
FROM rocm/dev-ubuntu-22.04:6.2.4@sha256:<verified-digest>
# Verify digest: docker pull rocm/dev-ubuntu-22.04:6.2.4 && docker inspect --format='{{index .RepoDigests 0}}' rocm/dev-ubuntu-22.04:6.2.4
```

**Running as root — add non-root USER**
```dockerfile
# AFTER — add near end of Dockerfile
RUN groupadd -r rocmuser && useradd -r -g rocmuser rocmuser
USER rocmuser
```

**ADD URL → COPY + verified download**
```dockerfile
# BEFORE (no integrity check)
ADD https://example.com/tool.tar.gz /tmp/

# AFTER
RUN curl -fsSL https://example.com/tool.tar.gz -o /tmp/tool.tar.gz \
    && echo "<sha256sum>  /tmp/tool.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/tool.tar.gz -C /opt/tool \
    && rm /tmp/tool.tar.gz
```

**RUN curl | bash → verified install**
```dockerfile
# BEFORE (arbitrary code execution at build time)
RUN curl https://install.example.com | bash

# AFTER (download, verify, then execute)
RUN curl -fsSL https://install.example.com/install.sh -o /tmp/install.sh \
    && echo "<sha256>  /tmp/install.sh" | sha256sum -c - \
    && chmod +x /tmp/install.sh && /tmp/install.sh \
    && rm /tmp/install.sh
```

**Missing no-new-privileges in docker-compose**
```yaml
# AFTER
services:
  rocm-worker:
    security_opt:
      - no-new-privileges:true
    read_only: true
    user: "1000:1000"
    cap_drop:
      - ALL
    cap_add:
      - SYS_PTRACE   # only if strictly required for ROCm debugger
```

**docker.sock volume mount**
```yaml
# BEFORE (full container escape path)
volumes:
  - /var/run/docker.sock:/var/run/docker.sock

# AFTER — use Docker-out-of-Docker proxy (e.g. nestybox/sysbox) or
# refactor to remove the Docker dependency inside the container entirely.
# If unavoidable, run the container as non-root and use a socket proxy
# (e.g. docker-socket-proxy) that restricts allowed API calls.
```

**Verify (R4):**
```bash
# Confirm the image does not run as root
docker inspect <image>:<tag> | jq '.[].Config.User'
# Expected: a non-empty, non-root value like "1000" or "rocmuser"

# Confirm no secrets are baked into image layers
docker history --no-trunc <image>:<tag> | grep -iE "secret|password|key|token"

# Confirm image is pinned to a digest
grep "FROM" Dockerfile | grep -v "@sha256:"
# Expected: zero output (all FROM lines have digest pins)

# Scan image for known CVEs
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image <image>:<tag>
```

**Prevent Regression (R4):**
```yaml
# .github/workflows/container-scan.yml
- name: Lint Dockerfile with Hadolint
  uses: hadolint/hadolint-action@<SHA>
  with:
    dockerfile: Dockerfile

- name: Trivy image scan
  uses: aquasecurity/trivy-action@<SHA>
  with:
    image-ref: ${{ env.IMAGE }}
    severity: HIGH,CRITICAL
    exit-code: 1
```

---

### R5 — IaC / Terraform / Kubernetes Remediations

**Public S3 bucket**
```hcl
# AFTER — block all public access
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

**Overly permissive IAM policy (Action: *, Resource: *)**
```hcl
# BEFORE
statement {
  actions   = ["*"]
  resources = ["*"]
  effect    = "Allow"
}

# AFTER — scope to minimum required actions and resources
statement {
  actions   = ["s3:GetObject", "s3:PutObject"]
  resources = ["arn:aws:s3:::my-rocm-artifacts/*"]
  effect    = "Allow"
}
```

**Unencrypted EBS / RDS**
```hcl
# AFTER
resource "aws_ebs_volume" "data" {
  encrypted  = true
  kms_key_id = aws_kms_key.ebs.arn
}

resource "aws_db_instance" "main" {
  storage_encrypted = true
  kms_key_id        = aws_kms_key.rds.arn
}
```

**Terraform state file committed**
```bash
# 1. Immediately rotate any credentials visible in the state file.
# 2. Remove: git rm --cached terraform.tfstate terraform.tfstate.backup
# 3. Add to .gitignore: echo "*.tfstate\n*.tfstate.backup" >> .gitignore
# 4. Configure remote encrypted backend:
terraform {
  backend "s3" {
    bucket         = "my-tf-state"
    key            = "rocm/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-east-1:123456789012:key/..."
    dynamodb_table = "tf-state-lock"
  }
}
```

**Kubernetes pod running as root / privileged**
```yaml
# AFTER — add securityContext to every container spec
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop:
      - ALL
```

**Kubernetes secret in plaintext YAML committed**
```bash
# 1. Rotate the exposed credential immediately.
# 2. Remove from repo: git rm --cached k8s/secret.yaml
# 3. Use Sealed Secrets:
#    kubeseal --format yaml < secret.yaml > sealed-secret.yaml
#    git add sealed-secret.yaml   # safe to commit — encrypted
# 4. Or use External Secrets Operator backed by AWS Secrets Manager / Vault.
```

**Kubernetes namespace without default-deny NetworkPolicy**
```yaml
# AFTER — apply to every namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: rocm-system
spec:
  podSelector: {}   # matches all pods in namespace
  policyTypes:
    - Ingress
    - Egress
  # No ingress/egress rules = deny all; add explicit allow rules as needed
```

---

### R6 — GPU / ROCm Runtime Remediations

**Missing hipDeviceSynchronize before host read**
```cpp
// BEFORE (host reads stale / incomplete device output)
hipLaunchKernelGGL(myKernel, grid, block, 0, 0, d_out);
float result;
hipMemcpy(&result, d_out, sizeof(float), hipMemcpyDeviceToHost);  // race!

// AFTER
hipLaunchKernelGGL(myKernel, grid, block, 0, 0, d_out);
HIP_CHECK(hipDeviceSynchronize());   // or use hipStreamSynchronize(stream)
HIP_CHECK(hipMemcpy(&result, d_out, sizeof(float), hipMemcpyDeviceToHost));
```

**Unvalidated kernel launch dimensions (integer overflow in grid/block)**
```cpp
// BEFORE
dim3 grid(user_n / BLOCK_SIZE);   // overflow if user_n is attacker-controlled

// AFTER
if (user_n == 0 || user_n > MAX_ELEMENTS)
    return hipErrorInvalidValue;
size_t grid_x = (user_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
if (grid_x > deviceProp.maxGridSize[0])
    return hipErrorInvalidValue;
dim3 grid(static_cast<unsigned>(grid_x));
```

**Unvalidated IPC handle from untrusted process**
```cpp
// BEFORE (arbitrary cross-process GPU memory access)
hipIpcMemHandle_t handle = receive_from_peer();
void *dev_ptr;
hipIpcOpenMemHandle(&dev_ptr, handle, hipIpcMemLazyEnablePeerAccess);

// AFTER — authenticate the peer before accepting its handle
if (!authenticate_peer(peer_pid, peer_uid))
    return -EPERM;
// Also: validate expected buffer size matches before use;
// restrict IPC to same-UID or privileged processes only
```

**env-var-controlled dlopen of plugin .so**
```cpp
// BEFORE (user can load arbitrary code via env var)
const char *path = getenv("ROCPROFILER_PLUGIN");
dlopen(path, RTLD_NOW);

// AFTER — validate against an allowlist of trusted directories
const char *path = getenv("ROCPROFILER_PLUGIN");
if (!path || !is_trusted_plugin_path(path))   // check realpath() prefix
    abort_with_error("Untrusted plugin path rejected");
// is_trusted_plugin_path: resolve realpath(), verify it starts with
// one of {"/opt/rocm/lib/rocprofiler/", "/usr/lib/rocprofiler/"}
// and that the file is owned by root with mode 0755 or stricter.
void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
```

**Missing __syncthreads() barrier in shared memory kernel**
```hip
// BEFORE (read before write completes — data race in LDS)
__shared__ float tile[TILE_SIZE];
tile[threadIdx.x] = input[idx];          // write
float val = tile[(threadIdx.x + 1) % TILE_SIZE];  // read — RACE

// AFTER
tile[threadIdx.x] = input[idx];
__syncthreads();   // barrier: all threads complete write before any read
float val = tile[(threadIdx.x + 1) % TILE_SIZE];
```

**Verify (R6):**
```bash
# Confirm hipDeviceSynchronize is present after kernel launches in the fixed file
grep -n "hipLaunchKernelGGL\|hipDeviceSynchronize\|hipStreamSynchronize" \
  path/to/fixed_file.cpp

# Run ThreadSanitizer (CPU host-side races)
cmake -DCMAKE_C_FLAGS="-fsanitize=thread" -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc) && ./tests/runtime_tests

# For GPU kernel races — build with ROCm's sanitizer:
/opt/rocm/bin/rocgdb --args ./your_binary
# Or: AMD GPU sanitizer (if available in your ROCm version):
HSA_TOOLS_LIB=/opt/rocm/lib/librocm-asan.so ./your_binary
```

**Prevent Regression (R6):**
```yaml
# Add GPU sanitizer build to CI matrix:
# .github/workflows/gpu-sanitize.yml
- name: GPU sanitizer run
  run: |
    cmake -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
          -DCMAKE_BUILD_TYPE=Sanitize ..
    make -j$(nproc)
    ctest --output-on-failure -L gpu_sanitize
  env:
    HSA_TOOLS_LIB: /opt/rocm/lib/librocm-asan.so
```

---

### R7 — Build Hardening Remediations

**Missing hardening flags in CMakeLists.txt**
```cmake
# AFTER — add to the top-level CMakeLists.txt (or a shared hardening module)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=2
    -Wformat
    -Wformat-security
    -Werror=format-security
    -fstack-clash-protection
    -fcf-protection=full
    -fvisibility=hidden        # for shared libraries
  )
  add_link_options(
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    -fPIE -pie                 # for executables
  )
endif()
```

**Hardening flags explicitly disabled**
```cmake
# REMOVE or guard these — they strip exploit mitigations:
# -fno-stack-protector          → remove
# -z execstack                  → remove
# -Wno-format-security          → remove
# -D_FORTIFY_SOURCE=0           → remove
# -O0 in release configs        → use -O2 or -O3 for release targets
```

**Verify (R7):**
```bash
# Confirm stack protector is present in the final binary
readelf -s ./build/lib/librocm_target.so | grep __stack_chk_fail
# Expected: at least one FUNC symbol named __stack_chk_fail

# Confirm RELRO and NX are enabled
checksec --file=./build/lib/librocm_target.so
# Expected: RELRO: Full, NX: enabled, Stack Canary: Yes, PIE: enabled

# Confirm _FORTIFY_SOURCE is active (look for fortified function variants)
nm ./build/lib/librocm_target.so | grep "__memcpy_chk\|__sprintf_chk"

# Confirm no -O0 in release build
cmake --build . --config Release -- VERBOSE=1 2>&1 | grep -E "\-O[0-9]"
```

**Prevent Regression (R7):**
```yaml
# .github/workflows/hardening-check.yml
- name: checksec binary audit
  run: |
    pip install checksec.py --break-system-packages
    checksec --file=./build/lib/librocm_target.so --output=json \
      | jq '.[] | select(.relro != "full" or .nx != true or .canary != true)'
    # Fail the job if any binary lacks full hardening

# CMakeLists.txt — fail the build if hardening flags are disabled
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  string(FIND "${CMAKE_C_FLAGS}" "-fno-stack-protector" _idx)
  if(NOT _idx EQUAL -1)
    message(FATAL_ERROR "Hardening flag -fno-stack-protector is prohibited in non-Debug builds")
  endif()
endif()
```

---

### R8 — Software Composition Analysis Remediations

**FetchContent / ExternalProject without URL_HASH**
```cmake
# BEFORE (no integrity check — poisoned tarball → build-time RCE)
FetchContent_Declare(msgpack
  URL https://github.com/msgpack/msgpack-c/archive/refs/tags/cpp-6.1.0.tar.gz
)

# AFTER (pinned hash)
FetchContent_Declare(msgpack
  URL https://github.com/msgpack/msgpack-c/archive/refs/tags/cpp-6.1.0.tar.gz
  URL_HASH SHA256=5e63e4d9b4a9f2e7a89dba29de4e1f2d734f5a3e...  # actual hash
)
```

**pip install without hash verification in CI**
```bash
# BEFORE
pip install -r requirements.txt

# AFTER (requires pip-compile --generate-hashes to produce the lock file)
pip install --require-hashes -r requirements.lock

# Or use a pinned lock file approach:
pip install pip-tools
pip-compile --generate-hashes requirements.in -o requirements.lock
```

**Unpinned or range-pinned Python dependency**
```toml
# BEFORE (pyproject.toml — unpinned)
[project]
dependencies = ["requests>=2.0", "PyYAML"]

# AFTER — pin in requirements.lock generated by pip-compile
# requirements.lock:
requests==2.32.3 \
    --hash=sha256:abcdef...
PyYAML==6.0.2 \
    --hash=sha256:123456...
```

**Dependency confusion risk (private package name on public registry)**
```bash
# Mitigation:
# 1. Register the internal package name on PyPI as a stub/placeholder
#    to prevent hijacking.
# 2. Use --index-url with your private registry and --no-index to prevent
#    fallback to PyPI for internal packages:
pip install --index-url https://my.private.registry/simple/ \
            --no-index internal-package
# 3. Or use namespace packages (com.amd.*) that are harder to squat.
```

**Missing SBOM**
```bash
# Generate CycloneDX SBOM for the component:
# Python: pip install cyclonedx-bom && cyclonedx-py environment -o sbom.json
# C++: cmake --build . --target cyclonedx-sbom  (if integrated)
# Or: syft dir:. -o cyclonedx-json > sbom.json
# Commit sbom.json and publish it with each release artifact.
```

**Verify (R8):**
```bash
# Confirm URL_HASH is present for every FetchContent_Declare
grep -A5 "FetchContent_Declare" CMakeLists.txt | grep -c "URL_HASH"
# Count must equal the number of FetchContent_Declare blocks

# Confirm pip install uses hashes in CI
grep -rn "pip install" .github/workflows/ | grep -v "require-hashes"
# Expected: zero output (all pip installs use --require-hashes or a lock file)

# Confirm no yanked packages in requirements
pip install pip-audit --break-system-packages
pip-audit -r requirements.lock

# Confirm SBOM exists and is non-empty
test -s sbom.json && echo "SBOM present" || echo "SBOM MISSING"
```

**Prevent Regression (R8):**
```yaml
# .github/workflows/sca.yml
- name: pip-audit dependency scan
  run: |
    pip install pip-audit --break-system-packages
    pip-audit -r requirements.lock --format=sarif -o pip-audit.sarif

- name: Upload pip-audit SARIF
  uses: github/codeql-action/upload-sarif@<SHA>
  with:
    sarif_file: pip-audit.sarif

# CMakeLists.txt — enforce URL_HASH on all FetchContent blocks
# Add a custom CMake function that wraps FetchContent_Declare and
# enforces URL_HASH presence:
function(rocm_fetch_content_declare name)
  cmake_parse_arguments(ARG "" "URL_HASH" "" ${ARGN})
  if(NOT ARG_URL_HASH)
    message(FATAL_ERROR "FetchContent_Declare for '${name}' must specify URL_HASH")
  endif()
  FetchContent_Declare(${name} ${ARGN})
endfunction()
```

---

## Output Format

### Report Delivery — MANDATORY

The calling script supplies an absolute path `REPORT_FILE`. You MUST persist the full report to that path — never print the report body to stdout.

1. As soon as the Coverage Map is built, `Write` sections 1–3 to `REPORT_FILE`.
2. After each subsequent section is complete, `Edit`-append it to `REPORT_FILE`. Do not buffer the whole report for one final Write — that causes truncation on large components.
3. When the report file is complete, your final stdout message must be exactly one line: `DONE <repo_name> <C>/<H>/<M>/<L>`. Nothing else.

If you exhaust budget mid-scan, the partially-written `REPORT_FILE` plus the Coverage Map already satisfies the "emit partial report" rule.

### Report Sections

Produce a single Markdown report with these sections:

1. **Executive summary** — component name, LOC, top 3 risks, overall posture. 
MUST include the following four subsections in order:

   **1a. Aggregate Metrics Dashboard** — two mandatory summary tables placed at the top of the executive summary so a reader can assess scope at a glance before reading any finding.

   | Severity | Count |
   |----------| ------|
   | Critical | # |
   | High     | # |
   | Medium   | # |
   | Low      | # |
   | Informational | # |
   | **Total**     | **#** |

   | Category | Count |
   |----------|-------|
   | Memory Safety     | # |
   | Injection / Shell | # |
   | Secret Exposure   | # |
   | Supply-Chain / SCA | # |
   | CI-CD / GitHub Actions | # |
   | Container       | # |
   | IaC / Cloud     | # |
   | GPU / Runtime   | # |
   | Build Hardening | # |
   | Trust Boundary  | # |
   | Other | # |

   **1b. Overall Posture Rating** — assign exactly one of the five ratings below. State the rating label, the specific criteria that triggered it, and a one-sentence plain-English justification. Do NOT leave this field as a vague adjective — it must map to the table.

   | Rating | Criteria |
   |--------|----------|
   | 🔴 **Critical** | Any attacker-reachable Critical finding OR ≥ 3 High findings that cross a trust boundary |
   | 🟠 **Poor** | No attacker-reachable Critical, but ≥ 1 Critical (local-only) OR ≥ 3 High findings |
   | 🟡 **Fair** | No Critical; 1–2 High findings; or predominately Medium findings with no High |
   | 🟢 **Good** | No Critical or High; Medium findings only; hardening gaps present but no exploitable bugs |
   | ✅ **Strong** | No Critical, High, or Medium findings; Low and hygiene issues only |

   Note: Informational findings do not affect the posture rating — they are counted in the dashboard but excluded from the rating criteria.

   Format: `**Overall Posture: 🟠 Poor** — 1 local-only Critical (stack overflow in loader, F-003) and 4 High findings crossing the user→kernel trust boundary.`

   **1c. Attack Surface Summary** — a table mapping every ROCm trust boundary to the count and highest severity of findings that cross it. List ALL boundaries from the ROCm-Specific Trust Boundaries section; write `None` in the Findings column if no finding touches that boundary. Any boundary with a Critical or High finding MUST include the representative finding ID.

   | Trust Boundary | Findings | Highest Severity | Representative Finding |
   |----------------|----------|------------------|------------------------|
   | KFD ioctl (user→kernel) | # | Critical / High / Medium / Low / None | ID — one-line title |
   | AQL packet / shared memory (user-writable) | # | | |
   | Code-object / ELF loader (hipModuleLoad\*, comgr) | # | | |
   | HIP/HSA env vars (env→process) | # | | |
   | rocm-smi / amdsmi (privileged sysfs) | # | | |
   | Doorbell / signal / event pages (mmap'd userspace) | # | | |
   | Plugin dlopen (env-controlled .so loading) | # | | |
   | CI/CD pipeline (fork PR → privileged runner) | # | | |
   | Build-time supply chain (FetchContent, pip, npm) | # | | |

   **1d. Compliance Framework Mapping** — a table showing how the findings map to common frameworks. Count only findings with Status `Open`. Write `0` if no findings map to a control — do not omit rows.

   | Framework | Control / Category | Open Findings |
   |-----------|--------------------|---------------|
   | CWE Top 25 (2023) | Any entry in the Top 25 list | # |
   | OWASP Top 10 (2021) | A03 Injection | # |
   | OWASP Top 10 (2021) | A06 Vulnerable & Outdated Components | # |
   | OWASP Top 10 (2021) | A09 Security Logging & Monitoring Failures | # |
   | NIST SP 800-53 | SI-10 Information Input Validation | # |
   | NIST SP 800-53 | SA-15 Development Process / Tools / Techniques | # |
   | NIST SP 800-53 | CM-6 Configuration Settings | # |
   | SLSA Level | Achieved level (0–3) based on build hardening and provenance findings | Level # |

   **1e. Quick Wins** — a numbered list of the top 10 actions sorted by impact-to-effort ratio (highest-severity + shortest fix time first). Format each line as: `<rank>. [<Severity>][<Effort>] <Ticket Summary> — <file:line>`.
2. **Scan metadata** — scan root, commit SHA, date, file/dir counts by language
3. **Coverage map** — table of EVERY top-level directory: `Directory | File count | Coverage (Full / Partial-pattern / None) | Notes` — any `None` entry MUST include a one-line justification
4. **Critical severity findings**
5. **High severity findings**
6. **Medium severity findings**
7. **Low severity findings**
8. **Informational findings** — style issues, `TODO`/`FIXME`/`HACK` comments, minor hygiene items. These do not affect the posture rating but must be listed for completeness.
9. **Memory-safety findings** (cross-referenced to 4–7)
10. **Trust-boundary / runtime isolation findings**
11. **Compiler / toolchain / build-system findings**
12. **Build hardening audit results** (table: `flag | present? | location`)
13. **Vendored & manifest dependency inventory** (table: `lib | current version | age | ecosystem | CVEs to verify | fix available? | fixed in version | EPSS% | integrity-verified?`). The **fixed in version** column must state the minimum upstream release that resolves each listed CVE, e.g. `zlib 1.3.1 (CVE-2023-45853)`. If no fix exists yet, write `no fix — mitigate by <workaround>`. The **EPSS%** column states the Exploit Prediction Scoring System probability for each CVE; write `N/A` for non-CVE entries.
14. **GitHub Actions / CI-CD findings**
15. **Container security findings** — every finding must include the introducing Dockerfile layer.
16. **IaC / Cloud / Kubernetes / runner findings**
17. **Secrets / credential exposure findings** — every finding must include the classified secret type.
18. **Security engineering gaps** (fuzzing, sanitizers, SBOM, signing, tests) — MUST include a **Repository Governance File Status** table showing `Present` / `Missing` for each of: `.gitignore`, `CODEOWNERS`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`. Any `Missing` entry is a Low-severity finding with a one-line remediation note.
19. **Pattern-sweep raw hits** (table: `file:line | matching line content | pattern | benign?` — include the actual matching line text, not just coordinates)
20. **Recommended fixes** — prioritized
21. **False-positive / low-confidence notes**
22. **Risk acceptance register** — list all findings with Status `Accepted Risk`, including owner, date, expiry, and justification. Write `None` if no findings are accepted.

---

## Per-Finding Template

```
### <ID> — <one-line title>

**File:** path/to/file.c:123-130
**Severity:** Critical | High | Medium | Low | Informational
**CVSS:** <base score> (<vector string>) — e.g. `8.8 (AV:L/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H)`. For CVE-backed findings use the NVD score. For code-only findings without a CVE, estimate from attack vector and impact and mark as `(estimated)`. Omit for Informational findings.
**Category:** Memory safety | Injection | Auth | Supply-chain | Secret exposure | Container | IaC | CI-CD | GPU / Runtime | Build Hardening | Governance | ...
**CWE:** CWE-### — <short CWE title, e.g. "CWE-121: Stack-based Buffer Overflow">
**Fix available:** Yes | No | Partial — one-line note, e.g. "upstream patched in zlib 1.3.1" or "no upstream fix; mitigate by removing the call".
**Status:** Open | Accepted Risk | False Positive | Resolved
**Trust boundary crossed:** user→kernel | file→parser | network→host | IPC | env→process | fork→CI | none
**Reachability:** attacker-reachable | local-only | requires-privilege | theoretical
**Exploitability:** Known exploit (PoC / in-the-wild) | No known exploit — for CVE-backed findings include EPSS score as `EPSS <score>%`; for code findings note whether a working exploit pattern is publicly documented (e.g. "standard heap-spray technique").
**Confidence:** High | Medium | Low
**Effort:** Minutes | Hours | Days | Redesign
**Introduced:** <commit SHA (first 8 chars)> by <author> on <YYYY-MM-DD> — obtain via `git log -S '<pattern>' --follow -- <file>` or `git blame -L <line>,<line> <file>`. Write `Unknown` if git history is unavailable.
**Secret type:** (secret exposure findings only) Classify as precisely as possible — e.g. AWS IAM Access Key | GitHub PAT | GCP Service Account Key | Azure Client Secret | PEM Private Key | NPM token | Generic high-entropy string.
**Introduced in layer:** (container findings only) Layer <N>: `<exact Dockerfile instruction>` — e.g. `Layer 4: RUN apt-get install libssl1.0-dev`.

**Ticket Summary:** One sentence suitable for a GitHub Issue or Jira title, e.g. "[Security][High] strcpy in hip_module_loader.cpp:247 allows stack overflow via attacker-controlled .hsaco file".
**Impact (plain English):** One or two sentences a non-security developer can understand — what could an attacker actually do? e.g. "An attacker who supplies a malformed .hsaco file can crash the ROCm runtime or run arbitrary code on the host machine."
**Description:** Technical explanation of what the code does and why it is unsafe.
**Evidence:** Minimal code excerpt (≤10 lines) with enough surrounding context to make the vulnerability self-evident without opening the file.
**Dependency chain:** (SCA / supply-chain findings only) Full transitive path from the direct dependency to the vulnerable package, e.g. `your-component → lib-a 2.1.0 → lib-b 1.4.3 (vulnerable, CVE-XXXX-YYYY)`. Write `N/A` for non-SCA findings.
**Exploit scenario:** Concrete path from untrusted input to impact, written for a security-aware reader.
**Remediation:** Specific, actionable fix. MUST include a unified diff (`diff -u` format) using the actual file path from the scan, or — for config/YAML changes — a before/after block. Do NOT write generic advice like "validate input" or "use a safe API". See the Remediation Guidance section for required example patterns per category.
**Verification:** Exact shell command(s) the developer can run after applying the fix to confirm it worked. See the Remediation Guidance section for examples per category.
**Prevent Regression:** The specific CI check, Semgrep rule, pre-commit hook, compiler flag, or test that prevents this class of issue from being reintroduced. See the Remediation Guidance section for examples per category.
**References:** Links to authoritative external sources. Include at minimum the CWE URL. Add NVD, OWASP, and vendor advisory links where applicable. Example: `[CWE-121](https://cwe.mitre.org/data/definitions/121.html) | [CVE-2023-45853 NVD](https://nvd.nist.gov/vuln/detail/CVE-2023-45853) | [OWASP Buffer Overflow](https://owasp.org/www-community/vulnerabilities/Buffer_Overflow)`.
**Taint path:** (SAST findings only) source → intermediate functions → sink, with file:line for each hop. Write `N/A` for non-SAST findings.
**Risk acceptance:** `None` — or if this finding is deliberately accepted: `Accepted by <owner> on <YYYY-MM-DD>, expires <YYYY-MM-DD>. Justification: <reason>`.
```

---

## Behavior

* Be exhaustive on coverage, conservative on claims.
* Prefer targeted deep analysis backed by the mandatory pattern sweep.
* Every finding must cite `file:line` and include all fields in the Per-Finding Template — no field may be omitted; write `N/A` when a field is not applicable to the finding type.
* Clearly separate exploitable vulnerabilities from hardening gaps from hygiene/style. Use the Informational severity level for style/hygiene items that carry no direct security risk.
* Do not report speculative issues unless explicitly marked `Confidence: Low`.
* Assign a CVSS v3.1 base score to every Critical, High, and Medium finding. For findings without a CVE use `(estimated)` after the vector string.
* For CVE-backed SCA findings, always include the EPSS score and state whether a fix version is available.
* For every confirmed finding, run `git blame` to populate the `Introduced` field. This is read-only; do not modify the repository.
* Deduplicate by root cause, not by symptom. If 15 call sites share the same root cause (e.g., no bounds-checking discipline in a module, or a single missing CMake flag that affects all targets), report ONE finding with the root cause as the title, list all affected locations in the Evidence field, and provide a single fix that resolves all of them. Do not emit 15 nearly-identical findings — that causes alert fatigue and obscures the actual fix.
* If you run out of budget mid-scan, emit the partial report with the Coverage Map showing exactly which directories remain `None` so a follow-up run can target them.
* For GitHub Actions findings, always quote the exact workflow file and line triggering the issue.
* For container findings, always specify the exact Dockerfile layer (instruction number and command) that introduces the risk in the `Introduced in layer` field.
* For IaC findings, always specify the resource type and attribute path (e.g., `aws_s3_bucket.artifacts.acl`).
* For secret findings, always redact the actual secret value in the report — report only the file path, line number, and classified secret type in the `Secret type` field.
* For SCA findings, always display the full transitive dependency chain from direct dependency to vulnerable package in the `Dependency chain` field.
* Populate the `References` field for every finding with at minimum the CWE URL. Add NVD and OWASP links where applicable.
* Set `Status: Open` on all new findings by default. Use `False Positive` only when the finding is demonstrably benign, with a one-line explanation in the `Risk acceptance` field.