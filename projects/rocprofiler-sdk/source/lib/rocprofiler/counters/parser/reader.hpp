#pragma ONCE

#include "parser.h"

// Bison functions for parsers
typedef struct yy_buffer_state* YY_BUFFER_STATE;
extern int
yyparse(rocprofiler::counters::RawAST** result);
extern YY_BUFFER_STATE
yy_scan_string(const char* str);
extern void
yy_delete_buffer(YY_BUFFER_STATE buffer);
