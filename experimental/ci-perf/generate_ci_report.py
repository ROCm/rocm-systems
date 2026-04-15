#!/usr/bin/env python3
"""Generate CI Analysis PDF Report for rocm-systems."""

from fpdf import FPDF
from datetime import datetime


class CIReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "ROCm Systems CI Analysis Report", align="L")
        self.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y')}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(20, 60, 120)
            self.ln(6)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(20, 60, 120)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)
        elif level == 2:
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(40, 80, 140)
            self.ln(4)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif level == 3:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(60, 100, 160)
            self.ln(3)
            self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(10)
        self.multi_cell(190, 5.5, f"{'':>{indent}}- {text}")

    def bold_bullet(self, label, text, indent=10):
        self.set_text_color(30, 30, 30)
        self.set_font("Helvetica", "", 10)
        self.set_x(10)
        self.multi_cell(190, 5.5, f"{'':>{indent}}- {label}: {text}")

    def code_block(self, text):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(245, 245, 248)
        self.set_text_color(40, 40, 40)
        self.ln(1)
        x = self.get_x()
        lines = text.split("\n")
        for line in lines:
            self.cell(190, 4.5, f"  {line}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 70, 130)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if self.get_y() > 265:
                self.add_page()
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(30, 70, 130)
                self.set_text_color(255, 255, 255)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
                self.ln()
                self.set_font("Helvetica", "", 8.5)
                self.set_text_color(30, 30, 30)
                fill = False

            if fill:
                self.set_fill_color(240, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 6, str(cell), border=1, fill=True, align=align)
            self.ln()
            fill = not fill
        self.ln(3)

    def key_finding(self, text):
        self.set_fill_color(255, 248, 230)
        self.set_draw_color(200, 170, 80)
        self.set_font("Helvetica", "I", 9.5)
        self.set_text_color(100, 70, 10)
        y = self.get_y()
        self.rect(12, y, 186, 8, style="DF")
        self.set_xy(15, y + 1.5)
        self.multi_cell(180, 5, text)
        self.ln(4)


def build_report():
    pdf = CIReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 15, "ROCm Systems", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "CI Architecture & Time Analysis", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(20, 60, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Repository: ROCm/rocm-systems", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Report Date: {datetime.now().strftime('%B %d, %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Branch: develop", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "Table of Contents", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    toc = [
        "1. Repository Overview",
        "2. CI Architecture - The Four CI Systems",
        "3. Code Quality & Formatting Pipelines",
        "4. Subtree Synchronization Pipelines",
        "5. Container / Docker Pipelines",
        "6. Testing Architecture",
        "7. Time Analysis - PR Wait Times",
        "8. Time Analysis - Nightly Builds",
        "9. Time Analysis - Per-Project Pipelines",
        "10. Media Libs CI - Deep Dive & Comparison",
        "11. Key Bottlenecks & Observations",
        "12. Historical Comparison: Feb / March / April 2026",
        "13. Media Libraries Dependency Map",
    ]
    for item in toc:
        pdf.cell(10, 6, "")
        pdf.cell(0, 6, item, new_x="LMARGIN", new_y="NEXT")

    # =========================================================================
    # 1. REPOSITORY OVERVIEW
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("1. Repository Overview")
    pdf.body_text(
        "rocm-systems is a git subtree-based monorepo consolidating 28+ ROCm projects under a single repository. "
        "CI is implemented primarily through GitHub Actions (~76 workflow files), with additional integration into "
        "Azure Pipelines and a legacy Jenkinsfile for HIP. The CI is heavily path-filtered -- most workflows only "
        "trigger when their specific project subtree is modified."
    )

    pdf.chapter_title("Project Structure", level=2)
    pdf.table(
        ["Directory", "Purpose", "Count"],
        [
            ["projects/", "ROCm component projects (each formerly its own repo)", "28"],
            ["shared/", "Shared dependencies used by multiple projects", "3"],
            ["experimental/", "Experimental / incubating projects", "3"],
            ["test/therock/", "Cross-project integration tests", "14 files"],
            [".github/workflows/", "CI workflow definitions", "76+"],
            [".github/scripts/", "CI automation Python scripts", "15"],
        ],
        col_widths=[42, 110, 38],
    )

    pdf.chapter_title("Key Projects", level=2)
    pdf.table(
        ["Group", "Projects"],
        [
            ["Core", "amdsmi, rocm-core, rocminfo, rocm-smi-lib"],
            ["Runtimes", "clr, hip, hip-tests, hipother, hotswap, rocr-runtime"],
            ["Profiler", "aqlprofile, rocprofiler-*, roctracer"],
            ["Debug Tools", "rocdbgapi, rocr-debug-agent"],
            ["Comm Libs", "rccl, rccl-tests, rocshmem"],
            ["Media Libs", "rocdecode, rocjpeg"],
            ["DC Tools", "rdc, cuid"],
        ],
        col_widths=[35, 155],
    )

    # =========================================================================
    # 2. CI ARCHITECTURE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("2. CI Architecture - The Four CI Systems")

    # 2.1 TheRock CI
    pdf.chapter_title("2.1 TheRock CI (Primary Build/Test Pipeline)", level=2)
    pdf.body_text(
        "The centerpiece of the CI, using the external TheRock build framework. It follows a layered "
        "reusable-workflow architecture with smart path-based routing."
    )

    pdf.chapter_title("Workflow Call Chain", level=3)
    pdf.code_block(
        "therock-ci.yml (orchestrator)\n"
        " +-- setup job\n"
        " |    +-- pr_detect_changed_subtrees.py  -> which subtrees changed?\n"
        " |    +-- therock_configure_ci.py        -> which projects to build/test?\n"
        " +-- therock-ci-linux.yml (reusable)\n"
        " |    +-- therock-test-packages.yml (reusable)\n"
        " |         +-- test_sanity_check\n"
        " |         +-- therock-test-component.yml (per-component, sharded)\n"
        " +-- therock-ci-windows.yml (reusable)\n"
        " |    +-- therock-test-packages.yml\n"
        " +-- therock-rccl-ci-linux.yml (reusable, RCCL-specific)\n"
        "      +-- therock-rccl-test-packages-single-node.yml\n"
        "      +-- therock-rccl-test-packages-multi-node.yml (Slurm, 4 nodes)"
    )

    pdf.chapter_title("Project Grouping (therock_matrix.py)", level=3)
    pdf.table(
        ["Group", "Subtrees", "Tests Run"],
        [
            ["core", "amdsmi, rocm-core, rocminfo, rocm-smi-lib", "sanity only"],
            ["runtimes", "clr, hip, hip-tests, hipother, hotswap, rocr-runtime", "hip-tests, rocrtst, rocprofiler-sdk"],
            ["profiler", "aqlprofile, rocprofiler-*, roctracer", "aqlprofile, rocprofiler-compute/sdk/systems"],
            ["debug_tools", "rocdbgapi, rocr-debug-agent", "rocr-debug-agent, rocgdb"],
            ["dc_tools", "rdc, cuid", "none (TBD)"],
            ["rocshmem", "rocshmem", "none (TBD)"],
            ["nightly", "all projects", "all + math libs (rocprim, rocblas, miopen...)"],
        ],
        col_widths=[28, 85, 77],
    )

    pdf.chapter_title("Trigger Behavior", level=3)
    pdf.table(
        ["Event", "Scope", "Platform"],
        [
            ["PR (opened/sync)", "Only affected project groups", "Linux; Windows if hip/clr/rocr changed"],
            ["Push to develop", "All mapped subtrees (full build)", "Linux only"],
            ["Nightly (7AM UTC)", "Full nightly matrix + math libs", "Linux + Windows"],
            ["workflow_dispatch", "User-specified projects", "Both"],
            ["Docs-only changes", "CI skipped entirely", "N/A"],
        ],
        col_widths=[38, 90, 62],
    )

    # 2.2 Azure Pipelines
    pdf.add_page()
    pdf.chapter_title("2.2 Azure Pipelines (External CI)", level=2)
    pdf.body_text(
        "A separate, public-facing CI system dispatched by the azure-ci-dispatcher.yml GitHub Action. "
        "It analyzes PR contents and triggers the appropriate Azure pipelines."
    )
    pdf.table(
        ["", "Ubuntu 22.04", "AlmaLinux 8"],
        [
            ["gfx942 build", "Supported", "Supported"],
            ["gfx90a build", "Supported", "Supported"],
            ["gfx942 test", "Supported", "Unsupported"],
            ["gfx90a test", "Supported", "Unsupported"],
            ["gfx1201/1100/1030", "In progress", "In progress"],
        ],
        col_widths=[55, 68, 67],
    )
    pdf.body_text(
        "Key feature: Downstream dependency triggers catch upstream breaking changes. "
        "Example: rocPRIM -> hipCUB + rocThrust; rocRAND -> hipRAND; hipBLAS-common -> hipBLASLt."
    )

    # 2.3 Per-Project CI
    pdf.chapter_title("2.3 Per-Project CI Workflows", level=2)
    pdf.table(
        ["Project", "Key Features"],
        [
            ["rocprofiler-sdk", "Multi-distro (Ubuntu/RHEL/SLES), sanitizers (ASan/TSan/LSan/UBSan), ROCm compat"],
            ["rocprofiler-systems", "Per-distro build/test, CI container builds, multi-GPU"],
            ["rocprofiler-compute", "Standalone + docker-compose testing, multi-distro"],
            ["aqlprofile", "Multi-GPU (navi4, navi3, mi325), multi-distro (deb + rpm)"],
            ["rocprofiler-register", "Multi-compiler (clang-13..15, gcc-11..12), coverage, sanitizers"],
            ["amdsmi", "9 Linux distro matrix, ABI backward compatibility checks"],
            ["rocr-runtime", "WSL build validation, std::filesystem ban enforcement"],
            ["media-libs", "Dedicated pipeline for rocdecode + rocjpeg (see Section 10)"],
        ],
        col_widths=[42, 148],
    )

    # 2.4 Internal ROCm CI
    pdf.chapter_title("2.4 Internal ROCm CI", level=2)
    pdf.body_text(
        "A private CI system triggered via rocm_ci_caller.yml on every PR to develop. "
        "Uses repository_dispatch to a private repo with a GitHub App token. "
        "Many subprojects also carry their own rocm_ci_caller.yml from their original repos."
    )

    # =========================================================================
    # 3. CODE QUALITY
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("3. Code Quality & Formatting Pipelines")
    pdf.body_text("Multiple independent formatting and linting workflows run on PRs:")
    pdf.table(
        ["Workflow", "Scope", "Tools", "Duration"],
        [
            ["hip-formatting", "hip, clr, hipother, hip-tests", "clang-format", "~1m"],
            ["rocprofiler-compute-formatting", "rocprofiler-compute", "ruff, gersemi, clang-format", "~6-7m"],
            ["rocprofiler-systems-formatting", "rocprofiler-systems", "markdownlint, spell, black, gersemi, clang-format", "~1m"],
            ["lstt-formatting", "rdc, amdsmi, rocm-smi-lib", "pre-commit hooks", "~1m"],
            ["rocprofiler-sdk-python", "rocprofiler-sdk", "flake8 (Py 3.8/3.10/3.12)", "~1m"],
            ["rocprofiler-sdk-restrictions", "rocprofiler-sdk", "grep ban on std::regex", "<1m"],
            ["rocr-runtime-restrictions", "rocr-runtime", "grep ban on std::filesystem", "<1m"],
            ["hip-validate-pr-description", "hip PRs", "template compliance check", "<1m"],
            ["ABI Compliance Check", "amdsmi", "abi-compliance-checker", "~2-3m"],
            ["CodeQL (aqlprofile/sdk)", "security", "CodeQL (Python + Actions)", "~5m"],
        ],
        col_widths=[52, 40, 58, 22],
    )

    # =========================================================================
    # 4. SUBTREE SYNC
    # =========================================================================
    pdf.chapter_title("4. Subtree Synchronization Pipelines")
    pdf.code_block(
        "Upstream -> Super-repo:\n"
        "  update-subtrees.yml (hourly cron) -> git subtree pull for each repo (~1m)\n"
        "\n"
        "Super-repo -> Downstream:\n"
        "  pr-merge-sync-patches.yml (on push to develop) -> detect changed\n"
        "  subtrees -> generate patches -> push to sub-repos (~1m)\n"
        "\n"
        "Sub-repo PR -> Super-repo:\n"
        "  pr-import.yml / import_pr_list.yml (manual) -> git subtree pull"
    )

    # =========================================================================
    # 5. CONTAINERS
    # =========================================================================
    pdf.chapter_title("5. Container / Docker Pipelines")
    pdf.table(
        ["Workflow", "Registry", "Schedule"],
        [
            ["rocprofiler-sdk-build-ci-docker-images", "docker.io/rocm/rocprofiler-private", "Daily"],
            ["rocprofiler-systems-containers", "Docker Hub", "Daily"],
            ["rocprofiler-systems-ghcr", "ghcr.io/rocm/rocprofiler-*", "Daily"],
            ["rocprofiler-ghcr-cleanup", "GHCR", "Daily (deletes images > 7 days)"],
        ],
        col_widths=[72, 75, 43],
    )

    # =========================================================================
    # 6. TESTING ARCHITECTURE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("6. Testing Architecture")

    pdf.chapter_title("Test Frameworks Used", level=2)
    pdf.table(
        ["Framework", "Usage", "Projects"],
        [
            ["CTest / GTest", "Primary C++ test framework", "rocprofiler-sdk, rocprofiler-register, rocr-runtime, rdc, kpack"],
            ["pytest", "Integration tests + Python bindings", "test/therock/, amdsmi, CI scripts"],
            ["Sanitizers", "ASan, TSan, LSan, UBSan", "rocprofiler-sdk, rocprofiler-register"],
            ["ABI compliance", "Backward compatibility", "amdsmi"],
            ["Code coverage", "gcov/llvm-cov (currently disabled)", "rocprofiler-sdk"],
            ["Multi-node GPU", "Slurm (4 nodes)", "RCCL"],
        ],
        col_widths=[35, 62, 93],
    )

    pdf.chapter_title("Test Execution Model", level=2)
    pdf.body_text(
        "TheRock tests follow a build-once, test-many model. Build artifacts are uploaded to AWS S3, "
        "then downloaded by test runners. Tests run on GPU-enabled runners with /dev/kfd and /dev/dri "
        "device access. Tests are sharded across multiple runners for parallelism."
    )
    pdf.body_text(
        "Three tiers of coverage: (1) PRs get targeted tests for changed subtrees only, "
        "(2) Nightly gets comprehensive tests including math libs, "
        "(3) Azure CI adds downstream dependency validation."
    )

    # =========================================================================
    # 7. TIME ANALYSIS - PR WAIT TIMES
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("7. Time Analysis - PR Wait Times")

    pdf.chapter_title("Profiler-Triggered PR (Typical Path)", level=2)
    pdf.table(
        ["Phase", "Duration", "Cumulative"],
        [
            ["Setup (detect subtrees, configure)", "~12s", "0m 12s"],
            ["Build Linux Packages", "~1h 31m", "1h 31m"],
            ["   Queue wait", "~3-6m", ""],
            ["   Checkout + fetch sources", "~5m", ""],
            ["   CMake configure", "~10m", ""],
            ["   Build (therock-archives/dist)", "~1h 10m", ""],
            ["   Upload artifacts to S3", "~5m", ""],
            ["Configure test matrix", "~20s", "1h 32m"],
            ["Test Sanity Check (1 shard)", "~2m", "1h 34m"],
            ["Test components (parallel):", "", "2h 12m"],
            ["   aqlprofile (1 shard)", "~1m", ""],
            ["   rocprofiler-sdk (1 shard)", "~6m", ""],
            ["   rocprofiler-systems (1 shard)", "~8m", ""],
            ["   rocprofiler-compute (2 shards)", "~33m", "SLOWEST"],
        ],
        col_widths=[80, 50, 50],
    )
    pdf.key_finding("Build is the bottleneck: ~65-75% of total PR CI time is spent in the TheRock Linux build.")

    pdf.chapter_title("Runtimes-Triggered PR (hip/clr/rocr changes)", level=2)
    pdf.table(
        ["Phase", "Duration"],
        [
            ["Build Linux Packages (ENABLE_ALL=ON)", "~2h 00m"],
            ["hip-tests (4 shards)", "~15m"],
            ["rocrtst (1 shard)", "~7m"],
            ["rocprofiler-sdk (1 shard)", "~9m"],
            ["rocprofiler-compute (2 shards)", "~30m"],
            ["TOTAL", "~2h 38m"],
        ],
        col_widths=[90, 90],
    )

    pdf.chapter_title("Developer Wait Time Summary", level=2)
    pdf.table(
        ["Change Area", "Total CI Wait", "Bottleneck"],
        [
            ["Profiler changes (rocprofiler-*)", "~2h 12m", "TheRock build (1h 31m)"],
            ["Runtime changes (hip, clr, rocr)", "~2h 38m", "TheRock build (2h) + tests (33m)"],
            ["Runtime + Windows (hip, clr)", "~3-4h", "Windows build + queue wait"],
            ["RCCL (single-node)", "~1h", "Build (49m)"],
            ["RCCL (multi-node)", "~4h", "Slurm queue (hours)"],
            ["AMDSMI changes", "~22m", "Parallel distro builds"],
            ["rocprofiler-sdk standalone", "~44m", "Full dependency build chain"],
            ["Core tools (rocminfo, smi)", "~22m", "Fast path, sanity tests only"],
            ["Media libs (rocdecode/rocjpeg)", "~38m", "Dedicated pipeline build"],
            ["Formatting only", "~1-7m", "Minimal"],
            ["Docs only", "~0m", "CI skipped entirely"],
        ],
        col_widths=[58, 42, 90],
    )

    # =========================================================================
    # 8. NIGHTLY BUILDS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("8. Time Analysis - Nightly Builds")

    pdf.chapter_title("Successful Nightly Run Breakdown (Apr 11, ~5h 30m)", level=2)
    pdf.table(
        ["Phase", "Duration", "Notes"],
        [
            ["Setup", "~10s", ""],
            ["Build Linux Packages", "1h 58m", "07:21 -> 09:18 UTC"],
            ["Queue wait for GPU runners", "2h 57m", "MAJOR BOTTLENECK"],
            ["Configure test matrix", "~11s", ""],
            ["sanity (1 shard)", "~1m", ""],
            ["aqlprofile (1 shard)", "~1m", ""],
            ["rocr-debug-agent (1 shard)", "~2m", ""],
            ["rocprofiler-systems (1 shard)", "~6m", ""],
            ["rocrtst (1 shard)", "~7m", ""],
            ["rocprofiler-sdk (1 shard)", "~9m", ""],
            ["hip-tests (4 shards)", "~15m", ""],
            ["rocgdb (1 shard)", "~30m", "Slowest test"],
            ["rocprofiler-compute (2 shards)", "~30m", ""],
        ],
        col_widths=[70, 40, 80],
    )
    pdf.key_finding("Nightly actual compute time is ~2.5-3h; the rest (~3h) is queue waiting for GPU runners.")

    pdf.chapter_title("Worst Case Nightly (Mar 27, ~10h with Windows)", level=2)
    pdf.table(
        ["Phase", "Duration"],
        [
            ["Linux Build", "~2h 03m"],
            ["Linux Tests", "~37m"],
            ["Windows Build", "~1h 39m"],
            ["Windows queue wait", "~2h 44m"],
            ["Windows hip-tests (4 shards, staggered)", "~4h+ (starvation)"],
            ["TOTAL", "~10h 00m"],
        ],
        col_widths=[90, 90],
    )

    pdf.chapter_title("Nightly Run History (Last 20 Runs)", level=2)
    pdf.table(
        ["Date", "Duration", "Result", "Notes"],
        [
            ["Apr 14", "3h 27m", "FAILURE", ""],
            ["Apr 13", "1h 01m", "SUCCESS", "Fastest recent run"],
            ["Apr 12", "7h 02m", "SUCCESS", ""],
            ["Apr 11", "5h 30m", "SUCCESS", ""],
            ["Apr 10", "5h 09m", "FAILURE", ""],
            ["Apr 09", "3h 59m", "FAILURE", ""],
            ["Apr 08", "2h 34m", "SUCCESS", ""],
            ["Apr 07", "4h 28m", "SUCCESS", ""],
            ["Apr 06", "2h 35m", "FAILURE", ""],
            ["Apr 05", "2h 43m", "SUCCESS", ""],
            ["Apr 04", "2h 40m", "FAILURE", ""],
            ["Apr 03", "3h 24m", "SUCCESS", ""],
            ["Apr 02", "3h 45m", "FAILURE", ""],
            ["Apr 01", "3h 15m", "FAILURE", ""],
            ["Mar 31", "3h 37m", "SUCCESS", ""],
            ["Mar 30", "4h 09m", "SUCCESS", ""],
            ["Mar 29", "4h 16m", "FAILURE", ""],
            ["Mar 28", "2h 40m", "SUCCESS", ""],
            ["Mar 27", "10h 00m", "FAILURE", "Windows queue starvation"],
            ["Mar 26", "2h 34m", "FAILURE", ""],
        ],
        col_widths=[30, 32, 32, 86],
    )
    pdf.key_finding("Nightly success rate over the last 20 runs: 50% (10/20). Average duration: ~3h 45m.")

    # =========================================================================
    # 9. PER-PROJECT TIMING
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("9. Time Analysis - Per-Project Pipelines")

    pdf.table(
        ["Pipeline", "Duration", "Jobs", "Notes"],
        [
            ["rocprofiler-sdk CI", "44m - 1h 07m", "4 distro matrix + sanitizers", "ubuntu-22.04 is longest (full dep build)"],
            ["AMDSMI CI", "~22m", "9 distro builds + 9 test suites", "All parallel, very efficient"],
            ["rocprofiler-systems CI", "13m - 1h 11m", "GHCR + standard CI", "12m typical; 1h for full distro"],
            ["AqlProfile CI", "20m - 52m", "3 GPU + 3 RPM distro", "~7m each, parallel"],
            ["rocprofiler-register CI", "~8m wall clock", "10 parallel matrix jobs", "Multi-compiler + sanitizers"],
            ["RocProf Trace Decoder", "~6-7m", "1 job (build + test)", "Lightweight"],
            ["Media Libs CI", "~38m", "Build + 2 tests", "See Section 10"],
            ["ROCR Runtime WSL", "4-13m", "1 job (WSL build)", "Variable WSL setup time"],
            ["kpack CI", "~4m", "Python pytest + C++ ctest", "Cross-platform"],
            ["RCCL CI (single-node)", "~1h", "Build + test", "gfx94X"],
            ["RCCL CI (multi-node)", "~4h", "Build + Slurm test", "gfx950, massive queue wait"],
            ["rocprofiler-compute CI", "~10h", "Extended test suite", "Frequently cancelled"],
        ],
        col_widths=[44, 30, 50, 66],
    )

    pdf.chapter_title("Infrastructure / Automation Timing", level=2)
    pdf.table(
        ["Workflow", "Duration", "Frequency"],
        [
            ["Synchronize Subtrees", "~1m", "Hourly"],
            ["Merged PR to Patch Subrepos", "~1m", "Every push to develop"],
            ["Trigger ROCm CI", "~13s", "Every PR (just dispatches)"],
            ["RTD Docs Sync", "~1m", "Push to develop with doc changes"],
            ["PR Org Label", "~1-2m", "Every 30 min"],
            ["Auto Label PR", "<1m", "Every PR event"],
            ["GHCR Cleanup", "~2-3m", "Daily"],
        ],
        col_widths=[65, 50, 75],
    )

    # =========================================================================
    # 10. MEDIA LIBS CI DEEP DIVE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("10. Media Libs CI - Deep Dive & Comparison")

    pdf.chapter_title("Pipeline Design", level=2)
    pdf.body_text(
        "The Media Libs CI (media-libs-ci.yml) is a standalone, stripped-down pipeline that runs "
        "instead of the main TheRock CI for media-only PRs. It exists because projects/rocdecode and "
        "projects/rocjpeg are commented out of therock_matrix.py (lines 14-15), carving media out of "
        "the main CI matrix entirely."
    )

    pdf.chapter_title("Build Scope Comparison", level=3)
    pdf.table(
        ["Flag", "Media Libs CI", "TheRock CI (profiler)"],
        [
            ["THEROCK_ENABLE_CORE", "ON", "ON (via ALL)"],
            ["THEROCK_ENABLE_ROCDECODE", "ON", "ON (via ALL)"],
            ["THEROCK_ENABLE_ROCJPEG", "ON", "ON (via ALL)"],
            ["THEROCK_ENABLE_SYSDEPS_AMD_MESA", "ON", "ON (via ALL)"],
            ["THEROCK_ENABLE_COMM_LIBS", "OFF", "ON"],
            ["THEROCK_ENABLE_DEBUG_TOOLS", "OFF", "ON"],
            ["THEROCK_ENABLE_MATH_LIBS", "OFF", "ON"],
            ["THEROCK_ENABLE_ML_LIBS", "OFF", "ON"],
            ["THEROCK_ENABLE_AQLPROFILE", "OFF", "ON"],
            ["THEROCK_ENABLE_ROCPROFV3", "OFF", "ON"],
            ["THEROCK_ENABLE_ROCPROFILER_COMPUTE", "OFF", "ON"],
            ["(20+ other subsystems)", "OFF", "ON"],
        ],
        col_widths=[72, 45, 50],
    )
    pdf.key_finding("Media pipeline disables 20+ subsystems, building only: core + AMD Mesa + rocdecode + rocjpeg.")

    pdf.chapter_title("Trigger Behavior", level=2)
    pdf.table(
        ["Event", "TheRock CI", "Media Libs CI"],
        [
            ["PR (media-only files)", "Setup only, skips all builds (~1m)", "Full media build + tests (~38m)"],
            ["PR (non-media files)", "Normal build + tests", "Not triggered"],
            ["Push to develop", "Full build of ALL subtrees (~2-3h)", "Also runs (~38m) -- OVERLAP"],
            ["Nightly", "Full nightly (no media tests)", "Not triggered"],
        ],
        col_widths=[48, 66, 76],
    )

    pdf.add_page()
    pdf.chapter_title("Build Time Statistics (10 Recent Runs)", level=2)
    pdf.table(
        ["Metric", "Media Libs CI Build", "TheRock CI Build (equiv.)", "Savings"],
        [
            ["Minimum", "27m", "1h 31m", "1h 04m (70%)"],
            ["Average", "35m", "1h 53m", "1h 18m (69%)"],
            ["Median", "37m", "1h 57m", "1h 20m (68%)"],
            ["Maximum", "41m", "2h 00m", "1h 19m (66%)"],
        ],
        col_widths=[40, 45, 55, 40],
    )

    pdf.chapter_title("Test Time Statistics", level=2)
    pdf.table(
        ["Metric", "Media Libs CI Tests", "TheRock CI Tests (equiv.)"],
        [
            ["Components tested", "2 (rocdecode, rocjpeg)", "4+ (profiler suite)"],
            ["Shards", "2 total (1 each)", "5+ (compute gets 2)"],
            ["Average duration", "~1.7m", "~15m"],
            ["GPU runner occupation", "~2m", "~33m (longest shard)"],
        ],
        col_widths=[48, 62, 70],
    )

    pdf.chapter_title("End-to-End Comparison", level=2)
    pdf.table(
        ["Metric", "Media Libs CI", "TheRock CI (equivalent)", "Savings"],
        [
            ["Build time (avg)", "~35m", "~1h 53m", "1h 18m (63%)"],
            ["Test time (avg)", "~2m", "~15m", "~13m"],
            ["Total (typical)", "~38m", "~2h 14m", "1h 36m (72%)"],
            ["Components built", "core + mesa + 2 libs", "entire ROCm stack", "~85% less"],
            ["GPU runner time", "~2m", "~33m", "~31m GPU saved"],
        ],
        col_widths=[42, 38, 55, 45],
    )
    pdf.key_finding("The media pipeline delivers 72% faster PR feedback: ~38m vs ~2h 14m for media developers.")

    pdf.chapter_title("Push-to-Develop Double Spend", level=2)
    pdf.body_text(
        "When a media PR merges to develop, BOTH pipelines run. The TheRock CI push path evaluates all "
        "mapped subtrees (since push events use the full subtree list), resulting in a full ~2-3h build. "
        "The Media Libs CI also runs its ~35m build in parallel."
    )
    pdf.body_text(
        "Real example -- '[rocdecode][rocjpeg] - update GPU targets' merge on Apr 13:"
    )
    pdf.table(
        ["Pipeline", "Build", "Queue", "Tests", "Total"],
        [
            ["Media Libs CI", "28m", "8m", "2m", "38m"],
            ["TheRock CI (push)", "2h 03m", "37m", "44m", "3h 24m"],
        ],
        col_widths=[40, 30, 30, 30, 40],
    )
    pdf.body_text(
        "The media build is redundant on push-to-develop. However, the media tests (rocdecode/rocjpeg) "
        "are NOT included in the TheRock CI push tests, so the media pipeline is the only place they "
        "get tested on merge -- which is valuable."
    )

    pdf.chapter_title("Nightly Coverage Gap", level=2)
    pdf.body_text(
        "Media libs are commented out of the nightly project_map as well. They are not tested in the "
        "comprehensive nightly run, making the media pipeline the sole CI path for these components. "
        "If the media pipeline has a blind spot, there is no second safety net."
    )

    # =========================================================================
    # 11. KEY BOTTLENECKS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("11. Key Bottlenecks & Observations")

    pdf.chapter_title("Critical Path Breakdown (where time goes)", level=2)
    pdf.code_block(
        "|||||||||||||||||||||||||||||||||||||||||||||||...............  TheRock Build   (~65-75%)\n"
        "...............................................|||||||||||||..  GPU Tests       (~15-20%)\n"
        "...............................................||||||||||||||  Setup / Upload  (~5-10%)"
    )

    pdf.chapter_title("Top 5 Bottlenecks", level=2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "1. TheRock Linux Build (1h 30m - 2h 00m)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "   The single largest time sink. Builds the entire ROCm stack from source in a container. "
        "Accounts for 65-75% of total PR CI time. ccache helps but the build is inherently large."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "2. GPU Test Runner Queue (0m - 6h)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "   Highly variable. Nightly and multi-node RCCL jobs frequently wait hours for GPU runners. "
        "The Apr 11 nightly spent 2h 57m queuing. The Mar 27 nightly took 10h total due to Windows "
        "GPU runner starvation."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "3. rocprofiler-compute Tests (~30m, 2 shards)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "   The slowest individual test component in the TheRock pipeline. Even with 2 shards, "
        "each shard runs ~28-33 minutes, forming the tail of the test phase."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "4. Windows Build + Queue (1h 39m build + hours queue)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "   When triggered (hip/clr/rocr changes), adds significant time. Nightly Windows runs "
        "suffer from test runner starvation where shards start at wildly different times."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "5. rocprofiler-compute Standalone CI (~10h)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "   The longest-running project-specific pipeline. Most recent runs are cancelled before "
        "completion due to new pushes superseding them."
    )

    pdf.chapter_title("Architectural Observations", level=2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Smart Path-Based Routing", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "The setup job in therock-ci.yml acts as a smart router. pr_detect_changed_subtrees.py "
        "identifies what changed, therock_configure_ci.py maps that to build flags and test lists, "
        "and only the relevant subset of the system gets built and tested. This prevents every PR "
        "from triggering a full 2h+ build."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Build-Once, Test-Many", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "TheRock builds produce artifacts uploaded to AWS S3, which are then downloaded by test jobs. "
        "This decouples build and test environments and allows GPU test runners to be different from "
        "build runners."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Media Pipeline: A Good Separation", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "The dedicated media pipeline is a successful example of carving out a fast-path for a "
        "self-contained component. It delivers 72% faster feedback for media developers while "
        "keeping the main pipeline uncluttered. The only inefficiency is a ~35m redundant build "
        "on push-to-develop, which is minor compared to the PR feedback gains."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Nightly Success Rate Needs Attention", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "Only 50% of nightly runs succeed (10/20 in the last 3 weeks). Failures are often caused by "
        "flaky tests or GPU runner availability rather than real regressions, which reduces trust in "
        "the nightly signal."
    )

    # =========================================================================
    # 12. HISTORICAL COMPARISON - FEB / MARCH / APRIL 2026
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("12. Historical Comparison: Feb / March / April 2026")
    pdf.body_text(
        "This section provides a 3-month trend analysis of CI run times from February through "
        "April 2026, tracking the rapid evolution of the CI infrastructure as new pipelines "
        "were added, build scopes expanded, and test coverage increased."
    )

    pdf.chapter_title("TheRock CI - PR Build Times (3-month trend)", level=2)
    pdf.table(
        ["Metric", "Feb 2026", "Mar 2026", "Apr 2026", "Trend"],
        [
            ["Profiler path (avg)", "~30m*", "130m", "107m", "30 -> 130 -> 107m"],
            ["Profiler path (min)", "22m", "126m", "91m", ""],
            ["Profiler path (max)", "41m", "133m", "118m", ""],
            ["Runtimes path (avg)", "N/A", "121m", "120m", "Stable at ~2h"],
        ],
        col_widths=[42, 30, 30, 30, 48],
    )
    pdf.body_text(
        "*February builds were dramatically faster because they used a smaller build scope "
        "(labeled 'hip-tests, rocprofiler-tests') rather than the current THEROCK_ENABLE_ALL=ON. "
        "The 4x increase from Feb to Mar reflects the expansion of build scope, not a regression."
    )
    pdf.key_finding(
        "PR builds: 30m (Feb) -> 130m (Mar) -> 107m (Apr). The Feb->Mar jump is due to "
        "expanded build scope; the Mar->Apr drop is genuine optimization."
    )

    pdf.chapter_title("Nightly Build Comparison", level=2)
    pdf.table(
        ["Metric", "Feb 2026", "Mar 2026", "Apr 2026"],
        [
            ["Status", "Did not exist", "Created Mar 6", "Fully active"],
            ["End-to-end (avg)", "N/A", "135m (2h 15m)", "231m (3h 51m)"],
            ["End-to-end (range)", "N/A", "67m - 207m", "61m - 422m"],
            ["Linux build time", "N/A", "123m avg", "118m"],
            ["Linux GPU tests", "N/A", "SKIPPED", "ACTIVE"],
            ["Success rate", "N/A", "87.5% (7/8)", "60% (6/10)"],
        ],
        col_widths=[40, 40, 50, 50],
    )
    pdf.body_text(
        "The TheRock CI Nightly workflow was created on March 6, 2026. It did not exist in "
        "February. March nightlies had GPU tests skipped; April enabled them, explaining the "
        "duration increase and lower success rate."
    )

    pdf.chapter_title("Critical Finding: Linux GPU Tests Enabled in April", level=2)
    pdf.set_fill_color(255, 235, 235)
    pdf.set_draw_color(200, 80, 80)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(150, 30, 30)
    y = pdf.get_y()
    pdf.rect(12, y, 186, 38, style="DF")
    pdf.set_xy(15, y + 2)
    pdf.multi_cell(180, 5, "MAJOR CHANGE DETECTED")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(80, 20, 20)
    pdf.set_x(15)
    pdf.multi_cell(180, 5,
        "In March, Linux GPU tests were SKIPPED on nearly all nightly runs. Only Windows "
        "hip-tests shards actually ran. In April, Linux GPU tests are fully active -- running "
        "hip-tests (4 shards), rocprofiler-sdk, rocprofiler-compute (2 shards), rocprofiler-systems, "
        "rocgdb, rocrtst, aqlprofile, and rocr-debug-agent.\n\n"
        "This explains: (1) Why April nightlies are longer (+96m avg), (2) Why April has a lower "
        "success rate (60% vs 87.5%) -- more tests means more failure surface, and (3) Why the duration "
        "variance is much wider in April (GPU runner queue contention)."
    )
    pdf.ln(6)

    pdf.chapter_title("Per-Project Pipeline Comparison (3-month)", level=2)
    pdf.table(
        ["Pipeline", "Feb 2026", "Mar 2026", "Apr 2026", "Trend"],
        [
            ["AMDSMI CI", "9m avg", "13m avg", "17m avg", "Steady growth (+89%)"],
            ["Media Libs CI", "No runs", "46m avg*", "42m avg*", "Stable (created ~Mar)"],
            ["AqlProfile CI", "14m avg", "19m avg", "15m avg", "Recovered after spike"],
            ["RCCL CI", "~1m (skip)", "79m", "38m avg", "Skip -> full -> optimized"],
            ["rocprofiler-systems", "24h+ timeout", "Cancelled", "13m typical", "Broken -> Fixed"],
            ["rocprofiler-sdk", "24h+ timeout", "44m-1h07m", "Similar", "Broken -> Fixed"],
        ],
        col_widths=[37, 27, 27, 27, 62],
    )
    pdf.body_text(
        "*Media Libs CI and RCCL CI had no actual builds in February -- workflows either did not "
        "exist or ran setup-only (skip). rocprofiler-systems and rocprofiler-sdk were broken with "
        "24h+ timeouts in February, fixed by March/April."
    )

    pdf.chapter_title("CI Infrastructure Evolution Timeline", level=2)
    pdf.table(
        ["Date", "Event", "Impact"],
        [
            ["Feb 2026", "TheRock CI uses smaller build scope", "PR builds ~30m (4x faster than current)"],
            ["Feb 2026", "rocprofiler-systems/sdk broken", "24h+ timeouts, all runs cancelled"],
            ["Feb 2026", "RCCL CI runs setup only", "No actual RCCL builds in CI"],
            ["Mar 6", "TheRock CI Nightly created", "First nightly runs (GPU tests skipped)"],
            ["Mar 2026", "Build scope expanded (ENABLE_ALL)", "PR builds jump to ~130m"],
            ["Mar 2026", "rocprofiler-systems fixed", "Down from 24h timeout to ~13m"],
            ["Late Mar", "Linux GPU tests enabled in nightly", "Nightly duration +71%, success -28pp"],
            ["Apr 2026", "Profiler builds optimized", "PR builds down to ~107m (-17%)"],
            ["Apr 2026", "RCCL CI optimized", "Down from 79m to ~38m (-52%)"],
        ],
        col_widths=[25, 70, 85],
    )

    pdf.chapter_title("Nightly Success Rate Trend", level=2)
    pdf.table(
        ["Period", "Runs", "Success", "Failure", "Pass Rate"],
        [
            ["Feb 2026", "N/A", "N/A", "N/A", "Nightly did not exist"],
            ["Mar 10-20", "8", "7", "1", "87.5%"],
            ["Mar 26 - Apr 4", "10", "5", "5", "50.0%"],
            ["Apr 5-14", "10", "6", "4", "60.0%"],
        ],
        col_widths=[38, 30, 30, 30, 42],
    )
    pdf.body_text(
        "The nightly success rate dropped from ~88% to ~55% as Linux GPU test coverage was enabled. "
        "This is not necessarily a regression -- it reflects increased test coverage catching real "
        "issues that were previously invisible."
    )

    pdf.chapter_title("Summary of 3-Month Trends", level=2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 100, 0)
    pdf.cell(0, 7, "Improvements (Feb -> Mar -> Apr):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.bullet("rocprofiler-systems CI: 24h+ timeout (Feb) -> cancelled (Mar) -> 13m (Apr) -- fully fixed")
    pdf.bullet("rocprofiler-sdk CI: 24h+ timeout (Feb) -> 44m-1h (Mar/Apr) -- fully fixed")
    pdf.bullet("RCCL CI: skip-only (Feb) -> 79m (Mar) -> 38m (Apr) -- 52% faster once active")
    pdf.bullet("Profiler PR builds: 130m (Mar) -> 107m (Apr) -- 17% faster")
    pdf.bullet("AqlProfile CI: 14m (Feb) -> 19m (Mar) -> 15m (Apr) -- recovered after March spike")
    pdf.bullet("Nightly pipeline created (Mar 6) and GPU test coverage enabled (late Mar/Apr)")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(180, 0, 0)
    pdf.cell(0, 7, "Regressions / Growth:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.bullet("PR build time: 30m (Feb) -> 107m (Apr) -- 3.5x increase due to expanded build scope (ENABLE_ALL)")
    pdf.bullet("AMDSMI CI: 9m (Feb) -> 13m (Mar) -> 17m (Apr) -- 89% growth, likely from distro matrix expansion")
    pdf.bullet("Nightly end-to-end: N/A -> 135m (Mar) -> 231m (Apr) -- +71% from enabled Linux GPU tests")
    pdf.bullet("Nightly success rate: N/A -> 87.5% (Mar) -> 60% (Apr) -- more coverage = more failure surface")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 7, "Key Takeaway:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.body_text(
        "The CI infrastructure has undergone rapid maturation over 3 months. February had minimal "
        "coverage with several broken pipelines. March introduced the nightly pipeline and expanded "
        "build scope. April enabled comprehensive GPU test coverage. The cost is longer build times "
        "and more failures, but the coverage and reliability improvements are substantial. The 30m "
        "February builds were fast because they built very little -- the current 107m reflects the "
        "true cost of building the full ROCm stack."
    )

    # =========================================================================
    # 13. MEDIA LIBRARIES DEPENDENCY MAP
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("13. Media Libraries Dependency Map")
    pdf.body_text(
        "This section maps the full dependency chain for rocdecode and rocjpeg, from the Linux "
        "kernel amdgpu driver up through the userspace graphics stack, ROCm runtime, and into "
        "the media libraries. Each layer builds on the one below it."
    )

    # Layer 0 - Hardware / Kernel
    pdf.chapter_title("Layer 0: Hardware & Kernel Driver", level=2)
    pdf.set_fill_color(220, 235, 255)
    pdf.set_draw_color(80, 120, 180)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 40, 80)
    y = pdf.get_y()
    pdf.rect(12, y, 186, 30, style="DF")
    pdf.set_xy(15, y + 2)
    pdf.multi_cell(180, 5,
        "AMD GPU Hardware (VCN Media Engines)\n"
        "  - VCN 2.5 (MI1xx/gfx908), VCN 2.6 (MI2xx/gfx90a), VCN 4.0 (MI3xx/gfx942, Navi3x)\n"
        "  - 1-4 VCN engines per GPU; H.264, H.265/HEVC, AV1, VP9, JPEG decode\n"
        "Linux Kernel: amdgpu KMD (kernel mode driver) + DRM subsystem"
    )
    pdf.ln(4)

    # Layer 1 - DRM userspace
    pdf.chapter_title("Layer 1: DRM Userspace (libdrm)", level=2)
    pdf.table(
        ["Component", "Provides", "Used By"],
        [
            ["libdrm", "DRM/KMS userspace API", "hsakmt, hsa-runtime64"],
            ["libdrm_amdgpu", "amdgpu device init, HW IP queries", "rocdecode, rocjpeg, hsakmt"],
        ],
        col_widths=[40, 75, 75],
    )
    pdf.body_text(
        "rocdecode uses amdgpu_query_hw_ip_count(AMDGPU_HW_IP_VCN_DEC) to discover available "
        "VCN decode engines. rocjpeg queries VCN JPEG capabilities. Both link libdrm_amdgpu directly."
    )

    # Layer 2 - VA-API
    pdf.chapter_title("Layer 2: VA-API Video Acceleration (libva + Mesa)", level=2)
    pdf.table(
        ["Component", "Provides", "Notes"],
        [
            ["AMD Mesa (radeonsi)", "VA-API backend driver for AMDGPU VCN", "Built by THEROCK_ENABLE_SYSDEPS_AMD_MESA=ON"],
            ["libva (>= 1.22)", "VA-API frontend: vaInitialize, vaCreateConfig, etc.", "Mandatory for both rocdecode and rocjpeg"],
            ["libva-drm", "DRM backend for VA-API (vaGetDisplayDRM)", "DMA-BUF surface export for HIP interop"],
        ],
        col_widths=[42, 80, 68],
    )
    pdf.body_text(
        "The VA-API layer is the bridge between the GPU's VCN hardware decoder and userspace. "
        "Mesa's radeonsi driver implements the VA-API interface for AMD GPUs. Both media libraries "
        "use VA-API to submit decode operations and export decoded surfaces as DMA-BUF handles."
    )

    # Layer 3 - ROCm Runtime Stack
    pdf.add_page()
    pdf.chapter_title("Layer 3: ROCm Runtime Stack (HSA / HIP)", level=2)
    pdf.table(
        ["Component", "Depends On", "Provides"],
        [
            ["libhsakmt (KMT thunk)", "libdrm, libdrm_amdgpu, libnuma, libelf", "Kernel mode thunk (KFD interface)"],
            ["hsa-runtime64 (ROCr)", "hsakmt, libdrm", "HSA Runtime API, device memory mgmt"],
            ["amd_comgr", "LLVM/amdclang", "Code object manager, ISA compilation"],
            ["CLR (rocclr + hipamd)", "hsa-runtime64, amd_comgr", "HIP implementation on AMD platform"],
            ["HIP Runtime", "CLR", "hip::host (CPU-side), hip::device (GPU kernels)"],
            ["rocprofiler-register", "hsa-runtime64", "API table registration for profiling"],
        ],
        col_widths=[45, 70, 75],
    )
    pdf.body_text(
        "TheRock CI enables this entire stack via -DTHEROCK_ENABLE_CORE=ON. HIP is the primary "
        "interface used by both media libraries for GPU memory management and compute kernel dispatch."
    )

    # Layer 4 - Media Libraries
    pdf.chapter_title("Layer 4: Media Libraries", level=2)
    pdf.table(
        ["Library", "HIP Target", "Mandatory Deps", "Optional Deps"],
        [
            ["rocdecode", "hip::host", "libva>=1.22, libva-drm, libdrm_amdgpu", "rocprofiler-register, FFmpeg"],
            ["rocjpeg", "hip::device", "libva>=1.22, libva-drm, libdrm_amdgpu", "rocprofiler-register"],
        ],
        col_widths=[30, 30, 75, 55],
    )
    pdf.body_text(
        "Key difference: rocdecode links hip::host (CPU-side HIP for external memory import), "
        "while rocjpeg links hip::device (GPU kernel compilation for color space conversion). "
        "Both are INDEPENDENT libraries -- rocjpeg does NOT depend on rocdecode."
    )

    # Layer 5 - Optional consumers
    pdf.chapter_title("Layer 5: Downstream Consumers (Optional)", level=2)
    pdf.table(
        ["Consumer", "Integration Type", "Purpose"],
        [
            ["rocdecode-host", "Sub-library of rocdecode", "FFmpeg-based host decoder (CPU fallback)"],
            ["rocprofiler-sdk", "FindrocDecode / FindrocJPEG", "API tracing/profiling support"],
            ["rocprofiler-systems", "Header-only (va/va.h)", "VAAPI call interception via GOTCHA"],
            ["Sample apps", "Links rocdecode/rocjpeg", "videoDecode, jpegDecode, etc."],
        ],
        col_widths=[40, 55, 95],
    )

    # Full visual dependency map
    pdf.add_page()
    pdf.chapter_title("Full Layered Dependency Map", level=2)
    pdf.body_text(
        "Complete dependency flow from hardware to application. Arrows (-->) indicate "
        "'depends on' relationships. Each layer builds on the one below."
    )
    pdf.code_block(
        "+=========================================================================+\n"
        "| LAYER 5: Applications & Profiling Tools                                 |\n"
        "|   Sample apps (videoDecode, jpegDecode, jpegDecodeBatched)              |\n"
        "|   rocprofiler-sdk (API tracing for rocdecode/rocjpeg)                   |\n"
        "|   rocprofiler-systems (VAAPI call interception)                         |\n"
        "+============================+==========================================+\n"
        "                             |                                            \n"
        "                             v                                            \n"
        "+=========================================================================+\n"
        "| LAYER 4: Media Libraries                                                |\n"
        "|                                                                         |\n"
        "|  +---------------------------+  +----------------------------+          |\n"
        "|  | rocdecode                 |  | rocjpeg                    |          |\n"
        "|  | - Video decode (H.264,    |  | - JPEG decode              |          |\n"
        "|  |   H.265, AV1, VP9)        |  | - Color space conversion   |          |\n"
        "|  | - Links: hip::host        |  | - Links: hip::device       |          |\n"
        "|  | - DMA-BUF import to HIP   |  | - DMA-BUF import to HIP   |          |\n"
        "|  +---------------------------+  +----------------------------+          |\n"
        "|  (Independent -- no cross-dependency between rocdecode & rocjpeg)       |\n"
        "+============================+==========================================+\n"
        "             |          |         |                                        \n"
        "             v          v         v                                        \n"
        "+=========================================================================+\n"
        "| LAYER 3: ROCm Runtime Stack         [THEROCK_ENABLE_CORE=ON]            |\n"
        "|                                                                         |\n"
        "|  HIP Runtime (hip::host / hip::device)                                  |\n"
        "|       |                                                                 |\n"
        "|  CLR (rocclr + hipamd)  +  amd_comgr (LLVM)                             |\n"
        "|       |                                                                 |\n"
        "|  hsa-runtime64 (ROCr)  +  rocprofiler-register (optional)               |\n"
        "|       |                                                                 |\n"
        "|  libhsakmt (KMT thunk to KFD)                                           |\n"
        "+============================+==========================================+\n"
        "             |          |         |                                        \n"
        "             v          v         v                                        \n"
        "+=========================================================================+\n"
        "| LAYER 2: VA-API Stack               [THEROCK_ENABLE_SYSDEPS_AMD_MESA=ON]|\n"
        "|                                                                         |\n"
        "|  libva (>= 1.22)  +  libva-drm                                          |\n"
        "|       |                                                                 |\n"
        "|  AMD Mesa (radeonsi VA-API driver for VCN)                               |\n"
        "+============================+==========================================+\n"
        "                             |                                            \n"
        "                             v                                            \n"
        "+=========================================================================+\n"
        "| LAYER 1: DRM Userspace                                                  |\n"
        "|                                                                         |\n"
        "|  libdrm  +  libdrm_amdgpu  (+libnuma, libelf for hsakmt)                |\n"
        "+============================+==========================================+\n"
        "                             |                                            \n"
        "                             v                                            \n"
        "+=========================================================================+\n"
        "| LAYER 0: Linux Kernel + Hardware                                        |\n"
        "|                                                                         |\n"
        "|  amdgpu KMD (kernel mode driver)  +  KFD (Kernel Fusion Driver)         |\n"
        "|  DRM subsystem  +  /dev/dri/renderD*  +  /dev/kfd                       |\n"
        "|                                                                         |\n"
        "|  AMD GPU Hardware: VCN 2.5-4.0 Media Engines (1-4 per GPU)              |\n"
        "|  Supported: MI1xx, MI2xx, MI3xx, Navi2x, Navi3x                         |\n"
        "+=========================================================================+"
    )

    # Cross-cutting dependencies table
    pdf.chapter_title("Cross-Cutting Dependencies", level=2)
    pdf.table(
        ["Dependency", "rocdecode", "rocjpeg", "Layer"],
        [
            ["amdgpu KMD + KFD", "Indirect (via HSA and Mesa)", "Indirect (via HSA and Mesa)", "0 - Kernel"],
            ["/dev/dri/renderD*", "Direct (DRM open)", "Direct (DRM open)", "0 - Kernel"],
            ["libdrm_amdgpu", "Direct link", "Direct link", "1 - DRM"],
            ["libva (>= 1.22)", "Direct link", "Direct link", "2 - VA-API"],
            ["libva-drm", "Direct link", "Direct link", "2 - VA-API"],
            ["Mesa radeonsi", "Runtime (VA-API driver)", "Runtime (VA-API driver)", "2 - VA-API"],
            ["HIP Runtime", "hip::host", "hip::device", "3 - ROCm"],
            ["hsa-runtime64", "Indirect (via HIP)", "Indirect (via HIP)", "3 - ROCm"],
            ["amd_comgr / LLVM", "Indirect (via HIP)", "Direct (GPU kernels)", "3 - ROCm"],
            ["rocprofiler-register", "Optional", "Optional", "3 - ROCm"],
            ["FFmpeg (avcodec/format)", "Optional (host decoder)", "Not used", "4 - Media"],
            ["amdclang++ (C++17)", "Build-time only", "Build-time only", "Toolchain"],
        ],
        col_widths=[42, 38, 38, 32],
    )

    # TheRock build integration
    pdf.add_page()
    pdf.chapter_title("TheRock Build System Integration", level=2)
    pdf.body_text(
        "The TheRock build system (external repo: ROCm/TheRock) orchestrates the full build. "
        "The following cmake flags control the media library build scope:"
    )
    pdf.table(
        ["CMake Flag", "What It Enables", "Required For"],
        [
            ["THEROCK_ENABLE_CORE=ON", "HSA, HIP, CLR, rocm-core, rocminfo, amdsmi", "Layer 3 (runtime stack)"],
            ["THEROCK_ENABLE_SYSDEPS_AMD_MESA=ON", "Mesa (radeonsi), libva, libdrm", "Layer 1-2 (DRM + VA-API)"],
            ["THEROCK_ENABLE_ROCDECODE=ON", "rocdecode library + tests", "Layer 4 (rocdecode)"],
            ["THEROCK_ENABLE_ROCJPEG=ON", "rocjpeg library + tests", "Layer 4 (rocjpeg)"],
            ["THEROCK_ENABLE_MEDIA_LIBS=ON", "Umbrella: rocdecode + rocjpeg", "Layer 4 (both)"],
        ],
        col_widths=[60, 68, 52],
    )
    pdf.body_text(
        "In the media-libs-ci.yml workflow, 20+ other subsystems (COMM_LIBS, DEBUG_TOOLS, "
        "MATH_LIBS, ML_LIBS, etc.) are explicitly set to OFF, producing a minimal build "
        "that only includes Layers 0-4 above."
    )

    pdf.chapter_title("TheRock sysdeps Detection", level=2)
    pdf.body_text(
        "Both rocdecode and rocjpeg detect TheRock builds by checking for the existence of "
        "${ROCM_PATH}/lib/rocm_sysdeps/lib. When detected, the Find modules for libva and "
        "libdrm_amdgpu search TheRock's sysdeps directory first, and RPATH is set to "
        "$ORIGIN/rocm_sysdeps/lib for bundled deployment."
    )

    pdf.chapter_title("VCN Hardware Capabilities", level=2)
    pdf.table(
        ["GPU Architecture", "VCN Gen", "VCN Count", "Video Codecs", "JPEG"],
        [
            ["gfx908 (MI1xx)", "2.5", "2", "H.264, H.265, VP9", "Yes"],
            ["gfx90a (MI2xx)", "2.6", "2", "H.264, H.265, VP9", "Yes"],
            ["gfx942 (MI3xx)", "4.0", "3-4", "H.264, H.265, AV1, VP9", "Yes"],
            ["gfx1030 (Navi2x)", "3.x", "2", "H.264, H.265, AV1, VP9", "Yes"],
            ["gfx1100 (Navi3x)", "4.0", "1-2", "H.264, H.265, AV1, VP9", "Yes"],
        ],
        col_widths=[38, 22, 25, 62, 20],
    )
    pdf.body_text(
        "rocdecode supports all VCN-equipped GPUs for video decode. rocjpeg uses the VCN JPEG "
        "decode engine. Multiple VCN engines per GPU enable parallel decode of multiple streams."
    )

    pdf.key_finding(
        "Media libs have a deep but narrow dependency chain: 5 layers from kernel to library, "
        "but only ~6 direct link dependencies. The key insight is that both rocdecode and rocjpeg "
        "bridge two worlds: the Linux graphics stack (DRM/VA-API/Mesa) and the ROCm compute stack "
        "(HSA/HIP), using DMA-BUF as the zero-copy interop mechanism."
    )

    return pdf


if __name__ == "__main__":
    pdf = build_report()
    output_path = "rocm-systems-ci-analysis.pdf"
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")
