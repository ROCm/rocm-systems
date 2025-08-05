%option noyywrap nodefault yylineno nounput

%{
#include <fmt/core.h>

#include "raw_ast.hpp"
#include "parser.h"
using namespace std;
#define YYDEBUG 1
%}

/* float exponent */
EXP	([Ee][-+]?[0-9]+)

%%
"+"	{ return ADD; }
"-"	{ return SUB; }
"*"	{ return MUL; }
"/"	{ return DIV; }
"|" { return ABS; }
"(" { return OP; }
")" { return CP; }
"=" { return EQUALS; }
"," { return CM; }
":" { return COLON; }
\[  { return O_SQ; }
\]  { return C_SQ; }

[0-9]+"."[0-9]*{EXP}? | 
"."?[0-9]+{EXP}? { 
    yylval.d = atoi(yytext);
    return NUMBER; }

"reduce" { return REDUCE; }
"select" { return SELECT; }
"accumulate" { return ACCUMULATE; }

[a-z_A-Z][a-z_A-Z0-9]*  { 
    yylval.s = strdup(yytext); 
    return NAME; }

([0-9]+(:[0-9]+)?)(,[0-9]+(:[0-9]+)?)* {
    yylval.s = strdup(yytext);
    return DIM_ARGS_RANGE;
}


\n      { return EOL; }
"//".*  
[ \t]   { /* ignore white space */ }
.	{ throw std::runtime_error(fmt::format("Mystery character {}", *yytext)); }
%%
