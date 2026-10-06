project(
    'addc-base',
    'c',
    'cpp',
    version: '3.3.0',
    default_options: [
        'cpp_std=c++20',
        'c_std=c11',
        'warning_level=3',
        'werror=true',
        'b_staticpic=true',
    ],
    meson_version: '>=1.3',
)

# ── Dependencies ───────────────────────────────────────────────────────────────

nlohmann_json_dep = dependency('nlohmann_json', required: true,
    fallback: ['nlohmann_json', 'nlohmann_json_dep'])

addc_version_str = meson.project_version()

# Product selection is the only build-time source of product truth. The generic
# CPER envelope is always present; private product section layouts and semantic
# routes are limited to this exact set.
base_all_projects = [
    'firerange', 'genoa', 'mi300a', 'mi300c', 'mi300x',
    'mi325x', 'mi350', 'mi450', 'turin', 'venice',
]
requested_projects = get_option('projects')
if requested_projects.length() == 0
    error('projects must contain at least one project or all')
endif
if 'all' in requested_projects
    if requested_projects.length() != 1
        error('projects=all cannot be combined with project names')
    endif
    selected_projects = base_all_projects
else
    selected_projects = requested_projects
    seen_projects = []
    foreach project_name : selected_projects
        if project_name not in base_all_projects
            error('unknown project in projects: ' + project_name)
        endif
        if project_name in seen_projects
            error('duplicate project in projects: ' + project_name)
        endif
        seen_projects += [project_name]
    endforeach
endif

enable_firerange = 'firerange' in selected_projects
enable_genoa = 'genoa' in selected_projects
enable_mi300a = 'mi300a' in selected_projects
enable_mi300c = 'mi300c' in selected_projects
enable_mi300x = 'mi300x' in selected_projects
enable_mi325x = 'mi325x' in selected_projects
enable_mi350 = 'mi350' in selected_projects
enable_mi450 = 'mi450' in selected_projects
enable_turin = 'turin' in selected_projects
enable_venice = 'venice' in selected_projects
enable_cpu_project = enable_firerange or enable_genoa or enable_turin or enable_venice
enable_mi300_gpu_project = enable_mi300a or enable_mi300c or enable_mi300x or enable_mi325x or enable_mi350

base_product_cpp_args = []
foreach project_name : base_all_projects
    if project_name in selected_projects
        base_product_cpp_args += '-DADDC_HAS_' + project_name.to_upper()
    endif
endforeach
test_product_cpp_args = base_product_cpp_args + [
    '-DADDC_TEST_PROJECT="' + selected_projects[0] + '"',
]
test_product_c_args = base_product_cpp_args + [
    '-DADDC_TEST_PROJECT="' + selected_projects[0] + '"',
]

# Header-only view of Base's supported consumer surface. Consumers receive
# these declarations without inheriting Base's private archives.
addc_public_headers_dep = declare_dependency(
    include_directories: include_directories('include'),
    dependencies: nlohmann_json_dep,
)

# Internal primitives shared by Base modules. These are implementation details,
# not a separate consumer-facing library.
addc_common_lib = static_library('addc_common',
    files('src/addc-common/base64.cpp'),
    include_directories: include_directories('src'),
)

addc_common_dep = declare_dependency(
    link_with: addc_common_lib,
    include_directories: include_directories('src'),
)

# Base owns the immutable identity and capability view. This internal
# dependency lets consumers reference its identities instead of copying them.
addc_product_identity_lib = static_library('addc_product_identity',
    files('src/products/product.cpp'),
    include_directories: include_directories('src/tier_api'),
    cpp_args: base_product_cpp_args,
)

addc_product_identity_dep = declare_dependency(
    link_with: addc_product_identity_lib,
    include_directories: include_directories('src/tier_api'),
)

# ── addc_cper_parser ───────────────────────────────────────────────────────────

addc_cper_parser_src = files(
    'src/addc-cper-parser/cper-parse.cpp',
    'src/addc-cper-parser/cper-record.cpp',
    'src/addc-cper-parser/section_dispatch.cpp',
    'src/addc-cper-parser/sections/cper-section-ainic.cpp',
    'src/addc-cper-parser/sections/cper-section-ia32x64.cpp',
    'src/addc-cper-parser/sections/cper-section-pcie.cpp',
)
if enable_cpu_project or enable_mi300_gpu_project
    addc_cper_parser_src += files(
        'src/addc-cper-parser/sections/cper-section-amd-epyc-crashdump.cpp',
    )
endif
if enable_cpu_project
    addc_cper_parser_src += files(
        'src/addc-cper-parser/sections/cper-section-amd-epyc-cdd.cpp',
        'src/addc-cper-parser/sections/cper-section-amd-epyc-pmic.cpp',
    )
endif
if enable_cpu_project or enable_mi450
    addc_cper_parser_src += files(
        'src/addc-cper-parser/sections/cper-section-amd-epyc-mtl.cpp',
    )
endif
if enable_mi300_gpu_project
    addc_cper_parser_src += files(
        'src/addc-cper-parser/sections/cper-section-amd-mi300-crashdump.cpp',
    )
endif
if enable_mi300_gpu_project or enable_mi450
    addc_cper_parser_src += files(
        'src/addc-cper-parser/sections/cper-section-amd-mi300-runtime.cpp',
    )
endif
if enable_mi450
    addc_cper_parser_src += files(
        'src/addc-cper-parser/sections/cper-section-amd-mi450-crashdump.cpp',
        'src/addc-cper-parser/sections/cper-section-amd-mi450-mtl.cpp',
    )
endif

addc_cper_parser_lib = static_library('addc_cper_parser', addc_cper_parser_src,
    include_directories: [include_directories('include'),
                          include_directories('src/tier_api')],
    dependencies: [nlohmann_json_dep, addc_common_dep],
    cpp_args: base_product_cpp_args,
)

addc_cper_parser_dep = declare_dependency(
    link_with: addc_cper_parser_lib,
    include_directories: [include_directories('include'),
                          include_directories('src/tier_api')],
    dependencies: [nlohmann_json_dep, addc_common_dep],
)

# ── addc_decoder ──────────────────────────────────────────────────────────────

addc_decoder_base_src = files(
    'src/addc-decoder/boot/boot_afid.cpp',
    'src/addc-decoder/dbglog/decoder.cpp',
    'src/addc-decoder/dbglog/venice.cpp',
    'src/addc-decoder/outbound/msg.cpp',
    'src/addc-decoder/pcie/decoder.cpp',
    'src/addc-decoder/wdt/decoder.cpp',
)
if enable_venice
    addc_decoder_base_src += files('src/addc-decoder/wdt/venice.cpp')
endif
if enable_mi450
    addc_decoder_base_src += files(
        'src/addc-decoder/boot_mi450/decoder.cpp',
        'src/addc-decoder/dbglog/mi450.cpp',
    )
endif

addc_decoder_base_lib = static_library('addc_decoder_base', addc_decoder_base_src,
    include_directories: [include_directories('include'),
                          include_directories('src/tier_api')],
    dependencies: [nlohmann_json_dep, addc_common_dep,
                   addc_product_identity_dep],
    cpp_args: base_product_cpp_args,
)

addc_decoder_base_dep = declare_dependency(
    link_with: addc_decoder_base_lib,
    include_directories: [include_directories('include'),
                          include_directories('src/tier_api')],
    dependencies: [nlohmann_json_dep, addc_common_dep,
                   addc_product_identity_dep],
)

addc_decoder_dep = addc_decoder_base_dep

# ── addc_mca ──────────────────────────────────────────────────────────────────

addc_mca_base_src = files(
    'src/addc-mca/decoder.cpp',
    'src/addc-mca/policy.cpp',
)
if enable_venice
    addc_mca_base_src += files('src/addc-mca/venice/venice.cpp')
endif
if enable_mi300x or enable_mi325x or enable_mi350
    addc_mca_base_src += files('src/addc-mca/mi300x.cpp')
endif
if enable_mi450
    addc_mca_base_src += files('src/addc-mca/mi450.cpp')
endif
if enable_mi300a
    addc_mca_base_src += files('src/addc-mca/mi300a.cpp')
endif

addc_mca_base_lib = static_library('addc_mca_base', addc_mca_base_src,
    include_directories: include_directories('include', 'src/tier_api'),
    dependencies: [nlohmann_json_dep, addc_product_identity_dep],
    cpp_args: ['-DADDC_VERSION="' + addc_version_str + '"'] + base_product_cpp_args,
)

addc_mca_base_dep = declare_dependency(
    link_with: addc_mca_base_lib,
    include_directories: include_directories('include', 'src/tier_api'),
    dependencies: [nlohmann_json_dep, addc_product_identity_dep],
)

addc_mca_dep = addc_mca_base_dep

# ── addc_cper_pipeline ────────────────────────────────────────────────────────

addc_cper_pipeline_src = files(
    'src/addc-cper-pipeline/detect.cpp',
    'src/addc-cper-pipeline/pipeline.cpp',
    'src/addc-cper-pipeline/section_dispatch.cpp',
    'src/addc-cper-parser/structural.cpp',
    'src/addc-cper-pipeline/sections/ainic_decoder.cpp',
    'src/addc-cper-pipeline/sections/ainic_section.cpp',
    'src/addc-cper-pipeline/sections/epyc_crashdump_section.cpp',
    'src/addc-cper-pipeline/sections/mtl_section.cpp',
    'src/addc-cper-pipeline/sections/mi450_mtl_section.cpp',
    'src/addc-cper-pipeline/sections/cdd_section.cpp',
    'src/addc-cper-pipeline/sections/pmic_section.cpp',
    'src/addc-cper-pipeline/sections/gpu_crashdump_section.cpp',
    'src/addc-cper-pipeline/sections/gpu_mi450_crashdump_section.cpp',
    'src/addc-cper-pipeline/sections/gpu_mi450_runtime_section.cpp',
    'src/addc-cper-pipeline/sections/gpu_runtime_section.cpp',
    'src/addc-cper-pipeline/sections/ia32x64_section.cpp',
    'src/addc-cper-pipeline/sections/pcie_section.cpp',
    'src/addc-cper-pipeline/sections/section_conversion.cpp',
)
if enable_cpu_project
    addc_cper_pipeline_src += files(
        'src/addc-cper-pipeline/sections/epyc_crashdump_decoder.cpp',
        'src/addc-cper-pipeline/sections/cdd_decoder.cpp',
        'src/addc-cper-pipeline/sections/pmic_decoder.cpp',
        'src/addc-cper-pipeline/sections/ia32x64_decoder.cpp',
        'src/addc-cper-pipeline/sections/pcie_decoder.cpp',
    )
endif
if enable_cpu_project or enable_mi450
    addc_cper_pipeline_src += files(
        'src/addc-cper-pipeline/sections/mtl_decoder.cpp',
    )
endif
if enable_mi300_gpu_project
    addc_cper_pipeline_src += files(
        'src/addc-cper-pipeline/sections/gpu_crashdump_decoder.cpp',
        'src/addc-cper-pipeline/sections/gpu_runtime_decoder.cpp',
    )
endif
if enable_mi450
    addc_cper_pipeline_src += files(
        'src/addc-cper-pipeline/sections/gpu_mi450_crashdump_decoder.cpp',
        'src/addc-cper-pipeline/sections/gpu_mi450_runtime_decoder.cpp',
    )
endif

addc_cper_pipeline_lib = static_library('addc_cper_pipeline', addc_cper_pipeline_src,
    include_directories: include_directories('include', 'src/tier_api'),
    dependencies: [nlohmann_json_dep, addc_common_dep, addc_cper_parser_dep,
                   addc_mca_base_dep, addc_decoder_base_dep],
    cpp_args: base_product_cpp_args,
)

addc_cper_pipeline_dep = declare_dependency(
    link_with: addc_cper_pipeline_lib,
    include_directories: include_directories('include', 'src/tier_api'),
    dependencies: [nlohmann_json_dep, addc_common_dep, addc_cper_parser_dep,
                   addc_mca_base_dep, addc_decoder_base_dep],
)

if get_option('enable_tests')
product_selection_test = executable(
    'product-selection-test',
    'tests/product_selection.cpp',
    dependencies: addc_mca_base_dep,
    cpp_args: '-DADDC_EXPECTED_PROJECTS="' + ','.join(selected_projects) + '"',
)
test('product-selection', product_selection_test)

base64_codec_test = executable(
    'base64-codec-test',
    'tests/base64_codec.cpp',
    dependencies: addc_common_dep,
)
test('base64-codec', base64_codec_test)

base64_fuzz_smoke = executable(
    'base64-fuzz-smoke',
    'fuzz/base64_decode_fuzz.cpp',
    dependencies: addc_common_dep,
    cpp_args: '-DADDC_FUZZ_SMOKE',
)
test('base64-fuzz-smoke', base64_fuzz_smoke)

cper_parser_fuzz_smoke = executable(
    'cper-parser-fuzz-smoke',
    'fuzz/cper_parser_fuzz.cpp',
    dependencies: addc_cper_parser_dep,
    cpp_args: '-DADDC_FUZZ_SMOKE',
)
test('cper-parser-fuzz-smoke', cper_parser_fuzz_smoke)

parser_modes_test = executable(
    'parser-modes-test',
    'tests/parser_modes.cpp',
    dependencies: addc_cper_parser_dep,
    cpp_args: base_product_cpp_args,
)
test('parser-modes', parser_modes_test)

parser_boundary_test = executable(
    'parser-boundary-test',
    'tests/parser_boundary.cpp',
    dependencies: addc_cper_pipeline_dep,
)
test('parser-boundary', parser_boundary_test)

allocation_contract_test = executable(
    'allocation-contract-test',
    'tests/allocation_contract.cpp',
    dependencies: [addc_common_dep, addc_cper_parser_dep,
                   addc_cper_pipeline_dep],
)
test('allocation-contract', allocation_contract_test)

if enable_mi300x and enable_mi450
    descriptor_platform_smoke = executable(
        'descriptor-platform-smoke',
        'tests/descriptor_platform_smoke.cpp',
        dependencies: addc_cper_pipeline_dep,
    )
    test('descriptor-platform-smoke', descriptor_platform_smoke)
endif

if enable_cpu_project
    section_dispatch_test = executable(
        'section-dispatch-test',
        'tests/section_dispatch.cpp',
        dependencies: addc_cper_pipeline_dep,
    )
    test('section-dispatch', section_dispatch_test)
endif

parser_classification_test = executable(
    'parser-classification-test',
    'tests/parser_classification.cpp',
    dependencies: addc_cper_parser_dep,
    cpp_args: base_product_cpp_args,
)
test('parser-classification', parser_classification_test)

pipeline_output_contract_test = executable(
    'pipeline-output-contract-test',
    'tests/pipeline_output_contract.cpp',
    dependencies: addc_cper_pipeline_dep,
)
test('pipeline-output-contract', pipeline_output_contract_test)

bad_page_policy_test = executable(
    'bad-page-policy-test',
    'tests/bad_page_policy.cpp',
    dependencies: addc_mca_base_dep,
)
test('bad-page-policy', bad_page_policy_test)

outbound_base_contract_test = executable(
    'outbound-base-contract-test',
    'tests/outbound_base_contract.cpp',
    dependencies: addc_decoder_base_dep,
)
test('outbound-base-contract', outbound_base_contract_test)

if enable_venice and enable_mi450 and enable_turin
    decoder_contract_test = executable(
        'decoder-contract-test',
        'tests/decoder_contract.cpp',
        dependencies: addc_decoder_base_dep,
    )
    test('decoder-contract', decoder_contract_test)
endif

if enable_mi450
    mca_contract_test = executable(
        'mca-contract-test',
        'tests/mca_contract.cpp',
        dependencies: addc_mca_base_dep,
    )
    test('mca-contract', mca_contract_test)
endif
endif

if get_option('enable_fuzzing')
    if meson.get_compiler('cpp').get_id() != 'clang'
        error('enable_fuzzing=true requires Clang')
    endif
    cper_parser_fuzz = executable(
        'cper-parser-fuzz',
        'fuzz/cper_parser_fuzz.cpp',
        dependencies: addc_cper_parser_dep,
        cpp_args: ['-fsanitize=fuzzer,address,undefined'],
        link_args: ['-fsanitize=fuzzer,address,undefined'],
    )
endif

# ── addc_lib ──────────────────────────────────────────────────────────────────

# Typed report resolution and summary projection are internal components used
# by Base's facade and dependent products. Keep them independent of the Base
# shared library so consumers need not link through a public API edge.
addc_base_report_lib = static_library('addc_base_report',
    files('src/core/base_report.cpp'),
    include_directories: include_directories('include', 'src/tier_api'),
    dependencies: [addc_cper_pipeline_dep, addc_public_headers_dep],
)

addc_base_report_dep = declare_dependency(
    link_with: addc_base_report_lib,
    include_directories: include_directories('include', 'src/tier_api'),
    dependencies: [addc_cper_pipeline_dep, addc_public_headers_dep],
)

if get_option('enable_cli') and not get_option('enable_lib')
    error('enable_cli=true requires enable_lib=true')
endif

if get_option('enable_lib')
addc_lib_link_args = []
if host_machine.system() == 'linux'
    # Keep symbols from static archives local to the shared library.
    addc_lib_link_args += '-Wl,--exclude-libs,ALL'
endif

addc_lib = library('addc_lib',
    files('src/api/lib.cpp', 'src/api/lib_c.cpp',
          'src/core/runtime.cpp'),
    include_directories: include_directories('include', 'src/tier_api'),
    dependencies: [addc_base_report_dep, addc_cper_parser_dep,
                   addc_cper_pipeline_dep, addc_mca_dep, addc_decoder_dep],
    gnu_symbol_visibility: 'hidden',
    c_args: ['-DADDC_BUILDING_LIB', '-DADDC_VERSION="' + addc_version_str + '"'],
    cpp_args: ['-DADDC_BUILDING_LIB', '-DADDC_VERSION="' + addc_version_str + '"'],
    link_args: addc_lib_link_args,
    version: addc_version_str,
    soversion: '1',
    install: true,
)

addc_lib_dep = declare_dependency(
    link_with: addc_lib,
    # Consumers link only the final facade. The parser/decoder archives stay
    # private; this dependency carries declarations and public header needs.
    dependencies: addc_public_headers_dep,
)

pkg = import('pkgconfig')
pkg.generate(addc_lib,
    name: 'addc-lib',
    description: 'addc-base CPER analysis library',
    requires: ['nlohmann_json'],
)

# ── addc-tool CLI ─────────────────────────────────────────────────────────────

if get_option('enable_cli')
addc_tool = executable(
    'addc-tool',
    'src/cli/main.cpp',
    include_directories: include_directories('src/tier_api'),
    dependencies: addc_lib_dep,
    cpp_args: ['-DADDC_VERSION="' + addc_version_str + '"'],
)
test('project-list-native-mca', addc_tool,
     args: ['mca', '--list-projects'])
endif

if get_option('enable_tests')
threads_dep = dependency('threads')

project_list_c_smoke = executable(
    'project-list-c-smoke',
    'tests/project_list_c_smoke.c',
    dependencies: addc_lib_dep,
    c_args: test_product_c_args,
)
test('project-list-c-abi', project_list_c_smoke)

cper_api_smoke = executable(
    'cper-api-smoke',
    'tests/cper_api_smoke.cpp',
    # This test compares the public facade with the legacy private pipeline;
    # declare those private test-only dependencies explicitly.
    dependencies: [addc_lib_dep, addc_cper_parser_dep,
                   addc_cper_pipeline_dep, addc_mca_base_dep,
                   addc_decoder_base_dep],
    cpp_args: test_product_cpp_args,
)
test('cper-api-contract', cper_api_smoke)

cper_c_abi_smoke = executable(
    'cper-c-abi-smoke',
    'tests/cper_c_abi_smoke.c',
    dependencies: addc_lib_dep,
    c_args: test_product_c_args,
)
test('cper-c-abi-contract', cper_c_abi_smoke)

contract_versions_test = executable(
    'contract-versions-test',
    'tests/contract_versions.cpp',
    dependencies: addc_lib_dep,
)
test('contract-versions', contract_versions_test)

context_concurrency_test = executable(
    'context-concurrency-test',
    'tests/context_concurrency.cpp',
    dependencies: [addc_lib_dep, threads_dep],
)
test('context-concurrency', context_concurrency_test)

endif
endif

if get_option('enable_install_headers')
    install_headers('include/addc/lib.hpp', subdir: 'addc')
    install_headers('include/addc/lib_c.h', subdir: 'addc')
    install_headers('include/addc/export.h', subdir: 'addc')
    install_headers('include/addc/options.hpp', subdir: 'addc')
    install_headers('include/addc/report_summary.hpp', subdir: 'addc')
    install_headers('include/addc/common.hpp', subdir: 'addc')
    install_headers('include/addc/json.hpp', subdir: 'addc')
    install_headers('include/addc/schema_version.hpp', subdir: 'addc')
endif
