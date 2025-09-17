/**
 * @file aql_arch_detect.c
 * @brief Architecture detection and operations table management
 *
 * This file implements the architecture detection logic and provides
 * access to architecture-specific operations tables. It allows runtime
 * selection of the appropriate architecture implementation based on
 * GPU identification strings.
 */

#include "aql_arch_ops.h"

#ifdef __KERNEL__
#include <linux/string.h>
#else
#include <string.h>
#include <stdio.h>
#endif

/*
 * Forward declarations for architecture operations tables
 * These are implemented in separate files (aql_gfx12_ops.c, etc.)
 */
extern const aql_arch_ops_t aql_gfx9_ops;
extern const aql_arch_ops_t aql_gfx12_ops;

/* TODO: Add other architectures as they are implemented */
/*
extern const aql_arch_ops_t aql_gfx10_ops;
extern const aql_arch_ops_t aql_gfx11_ops;
*/

/*
 * Architecture Detection Patterns
 *
 * Each entry contains patterns to match against GPU names and the
 * corresponding operations table.
 */
typedef struct {
    const char* pattern;                /**< Pattern to match in GPU name */
    const aql_arch_ops_t* ops;         /**< Operations table for this architecture */
} aql_arch_pattern_t;

static const aql_arch_pattern_t architecture_patterns[] = {
    /* GFX12 (RDNA3) patterns */
    { "gfx120",     &aql_gfx12_ops },
    { "gfx1200",    &aql_gfx12_ops },
    { "gfx1201",    &aql_gfx12_ops },
    { "rdna3",      &aql_gfx12_ops },

    /* GFX9 (Vega) patterns */
    { "gfx90",      &aql_gfx9_ops },
    { "gfx94",      &aql_gfx9_ops },
    { "gfx900",     &aql_gfx9_ops },
    { "gfx902",     &aql_gfx9_ops },
    { "gfx906",     &aql_gfx9_ops },
    { "gfx908",     &aql_gfx9_ops },
    { "gfx90a",     &aql_gfx9_ops },
    { "gfx940",     &aql_gfx9_ops },
    { "gfx942",     &aql_gfx9_ops },
    { "vega",       &aql_gfx9_ops },

    /* TODO: Add other architecture patterns */
    /*
    // GFX11 (RDNA2) patterns
    { "gfx110",     &aql_gfx11_ops },
    { "gfx1100",    &aql_gfx11_ops },
    { "gfx1101",    &aql_gfx11_ops },
    { "gfx1102",    &aql_gfx11_ops },
    { "rdna2",      &aql_gfx11_ops },

    // GFX10 (RDNA1) patterns
    { "gfx101",     &aql_gfx10_ops },
    { "gfx103",     &aql_gfx10_ops },
    { "gfx1010",    &aql_gfx10_ops },
    { "gfx1030",    &aql_gfx10_ops },
    { "rdna1",      &aql_gfx10_ops },
    { "gfx900",     &aql_gfx9_ops },
    { "gfx902",     &aql_gfx9_ops },
    { "gfx906",     &aql_gfx9_ops },
    { "gfx908",     &aql_gfx9_ops },
    { "gfx90a",     &aql_gfx9_ops },
    { "gfx940",     &aql_gfx9_ops },
    { "gfx942",     &aql_gfx9_ops },
    { "vega",       &aql_gfx9_ops },
    */

    /* Terminator */
    { NULL, NULL }
};

/*
 * Architecture Operations Table Registry
 */
static const aql_arch_ops_t* supported_architectures[] = {
    &aql_gfx9_ops,
    &aql_gfx12_ops,
    /* TODO: Add other architectures */
    /*
    &aql_gfx10_ops,
    &aql_gfx11_ops,
    */
    NULL
};

/*
 * Public API Functions
 */

const aql_arch_ops_t* aql_detect_architecture(const char* gfx_name) {
    const aql_arch_pattern_t* pattern;

    if (!gfx_name) {
        return NULL;
    }

    /* Convert to lowercase for case-insensitive matching */
    char lowercase_name[AQL_MAX_ARCH_NAME_LEN];
    size_t len = strlen(gfx_name);

    if (len >= sizeof(lowercase_name)) {
        len = sizeof(lowercase_name) - 1;
    }

    for (size_t i = 0; i < len; i++) {
        char c = gfx_name[i];
        if (c >= 'A' && c <= 'Z') {
            c = c - 'A' + 'a';
        }
        lowercase_name[i] = c;
    }
    lowercase_name[len] = '\0';

    /* Search for matching pattern */
    for (pattern = architecture_patterns; pattern->pattern != NULL; pattern++) {
        if (strstr(lowercase_name, pattern->pattern) != NULL) {
            return pattern->ops;
        }
    }

    return NULL; /* No matching architecture found */
}

const aql_arch_ops_t* aql_get_arch_ops(aql_arch_type_t arch_type) {
    const aql_arch_ops_t* const* ops_ptr;

    for (ops_ptr = supported_architectures; *ops_ptr != NULL; ops_ptr++) {
        if ((*ops_ptr)->arch_type == arch_type) {
            return *ops_ptr;
        }
    }

    return NULL;
}

uint32_t aql_list_supported_architectures(const aql_arch_ops_t** ops_list,
                                         uint32_t max_count) {
    const aql_arch_ops_t* const* src_ptr;
    uint32_t count = 0;

    if (!ops_list || max_count == 0) {
        return 0;
    }

    for (src_ptr = supported_architectures; *src_ptr != NULL && count < max_count; src_ptr++) {
        ops_list[count] = *src_ptr;
        count++;
    }

    return count;
}

/*
 * Architecture Information Query Functions
 */

/**
 * @brief Get architecture type from name string
 * @param gfx_name GPU architecture name
 * @return Architecture type, or AQL_ARCH_UNKNOWN if not recognized
 */
aql_arch_type_t aql_detect_arch_type(const char* gfx_name) {
    const aql_arch_ops_t* ops = aql_detect_architecture(gfx_name);
    return ops ? ops->arch_type : AQL_ARCH_UNKNOWN;
}

/**
 * @brief Check if architecture is supported
 * @param gfx_name GPU architecture name
 * @return True if architecture is supported
 */
bool aql_is_architecture_supported(const char* gfx_name) {
    return aql_detect_architecture(gfx_name) != NULL;
}

/**
 * @brief Get architecture capabilities
 * @param gfx_name GPU architecture name
 * @param capabilities Output structure for capabilities
 * @return AQL_SUCCESS if architecture is supported
 */
aql_result_t aql_get_arch_capabilities(const char* gfx_name,
                                      aql_arch_capabilities_t* capabilities) {
    const aql_arch_ops_t* ops;

    if (!gfx_name || !capabilities) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    ops = aql_detect_architecture(gfx_name);
    if (!ops) {
        return AQL_ERROR_UNSUPPORTED_ARCH;
    }

    /* Copy capabilities */
    capabilities->has_pred_exec = ops->has_pred_exec;
    capabilities->has_uconfig_space = ops->has_uconfig_space;
    capabilities->has_dual_sdma = ops->has_dual_sdma;
    capabilities->has_gl_cache_hierarchy = ops->has_gl_cache_hierarchy;
    capabilities->has_smn_addressing = ops->has_smn_addressing;

    return AQL_SUCCESS;
}

/**
 * @brief Get register space information for architecture
 * @param gfx_name GPU architecture name
 * @param register_spaces Output structure for register spaces
 * @return AQL_SUCCESS if architecture is supported
 */
aql_result_t aql_get_register_spaces(const char* gfx_name,
                                    aql_register_spaces_t* register_spaces) {
    const aql_arch_ops_t* ops;

    if (!gfx_name || !register_spaces) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    ops = aql_detect_architecture(gfx_name);
    if (!ops) {
        return AQL_ERROR_UNSUPPORTED_ARCH;
    }

    /* Copy register space information */
    *register_spaces = ops->register_spaces;

    return AQL_SUCCESS;
}

/*
 * Debug and Information Functions
 */

/**
 * @brief Print information about all supported architectures
 */
void aql_print_supported_architectures(void) {
    const aql_arch_ops_t* ops_list[16];
    uint32_t count;

    count = aql_list_supported_architectures(ops_list, 16);

#ifdef __KERNEL__
    printk(KERN_INFO "AQL: Supported architectures (%u):\n", count);
    for (uint32_t i = 0; i < count; i++) {
        printk(KERN_INFO "AQL:   %s (GFX%u) - %s%s%s%s%s\n",
               ops_list[i]->arch_name,
               ops_list[i]->gfx_version,
               ops_list[i]->has_pred_exec ? "PRED_EXEC " : "",
               ops_list[i]->has_uconfig_space ? "UCONFIG " : "",
               ops_list[i]->has_dual_sdma ? "DUAL_SDMA " : "",
               ops_list[i]->has_gl_cache_hierarchy ? "GL_CACHE " : "",
               ops_list[i]->has_smn_addressing ? "SMN " : "");
    }
#else
    printf("AQL: Supported architectures (%u):\n", count);
    for (uint32_t i = 0; i < count; i++) {
        printf("AQL:   %s (GFX%u) - %s%s%s%s%s\n",
               ops_list[i]->arch_name,
               ops_list[i]->gfx_version,
               ops_list[i]->has_pred_exec ? "PRED_EXEC " : "",
               ops_list[i]->has_uconfig_space ? "UCONFIG " : "",
               ops_list[i]->has_dual_sdma ? "DUAL_SDMA " : "",
               ops_list[i]->has_gl_cache_hierarchy ? "GL_CACHE " : "",
               ops_list[i]->has_smn_addressing ? "SMN " : "");
    }
#endif
}

/**
 * @brief Get architecture information string
 * @param gfx_name GPU architecture name
 * @param info_str Output buffer for information string
 * @param max_len Maximum length of output buffer
 * @return AQL_SUCCESS if architecture is supported
 */
aql_result_t aql_get_arch_info_string(const char* gfx_name, char* info_str, size_t max_len) {
    const aql_arch_ops_t* ops;
    int ret;

    if (!gfx_name || !info_str || max_len == 0) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    ops = aql_detect_architecture(gfx_name);
    if (!ops) {
        return AQL_ERROR_UNSUPPORTED_ARCH;
    }

#ifdef __KERNEL__
    ret = snprintf(info_str, max_len, "%s (GFX%u): %s%s%s%s%s",
                   ops->arch_name, ops->gfx_version,
                   ops->has_pred_exec ? "PRED_EXEC " : "",
                   ops->has_uconfig_space ? "UCONFIG " : "",
                   ops->has_dual_sdma ? "DUAL_SDMA " : "",
                   ops->has_gl_cache_hierarchy ? "GL_CACHE " : "",
                   ops->has_smn_addressing ? "SMN " : "");
#else
    ret = snprintf(info_str, max_len, "%s (GFX%u): %s%s%s%s%s",
                   ops->arch_name, ops->gfx_version,
                   ops->has_pred_exec ? "PRED_EXEC " : "",
                   ops->has_uconfig_space ? "UCONFIG " : "",
                   ops->has_dual_sdma ? "DUAL_SDMA " : "",
                   ops->has_gl_cache_hierarchy ? "GL_CACHE " : "",
                   ops->has_smn_addressing ? "SMN " : "");
#endif

    if (ret < 0 || (size_t)ret >= max_len) {
        return AQL_ERROR_BUFFER_TOO_SMALL;
    }

    return AQL_SUCCESS;
}