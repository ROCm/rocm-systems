/**
 * @file arch_creator.h
 * @brief Architecture creator interface for GPU architectures
 */

#ifndef ARCH_CREATOR_H
#define ARCH_CREATOR_H

#include "aql_structures.h"

/* Architecture creation functions */
arch_t* arch_create_by_name(const char* arch_name);
arch_t* arch_create_gfx12(void);
void arch_destroy(arch_t* arch);

/* Architecture initialization */
int arch_init(arch_t* arch, arch_type_t type);

#endif /* ARCH_CREATOR_H */