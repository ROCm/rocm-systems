#ifndef ERROR_MAP_H
#define ERROR_MAP_H

#include <stdint.h>

/**
 * @brief Structure representing an error mapping entry
 */
typedef struct
{
    uint32_t id;
    const char *error_category;
    const char *error_type;
    const char *method;
    const char *error_severity;
} error_map_entry_t;

/**
 * @brief Get error ID based on category, type and severity
 * @param[in] error_category Error category string
 * @param[in] error_type Error type string
 * @param[in] error_severity Error severity string
 * @return Error ID if found, -1 if not found
 */
int get_error_id(const char *error_category, const char *error_type, const char *error_severity);

#endif /* ERROR_MAP_H */
