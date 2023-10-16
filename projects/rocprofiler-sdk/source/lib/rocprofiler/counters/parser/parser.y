%parse-param {RawAST** result}
%code requires {
#include "raw_ast.hpp"
using namespace rocprofiler::counters;
#define YYDEBUG 1
}

%{
#include <stdexcept>
#include <stdio.h>
#include <string>

#include <glog/logging.h>

#include "raw_ast.hpp"

int yyparse(rocprofiler::counters::RawAST** result);
int yylex(void);
void yyerror(rocprofiler::counters::RawAST**, const char *s) { LOG(ERROR) << s; }
%}

/* declare tokens */
%token ADD SUB MUL DIV ABS EQUALS
%token OP CP O_SQ C_SQ COLON
%token EOL

/* set associativity rules for operand tokens */
%right EQUALS
%left ADD SUB
%left MUL DIV
%nonassoc '|' UMINUS CM

/*declare data types*/
%union {
 RawAST* a;          /* For ast node */
 int64_t d;
 char* s;
}

%token NUMBER RANGE              /* set data type for numbers */
%token NAME                      /* set data type for variables and user-defined functions */
%token REDUCE SELECT             /* set data type for special functions */
%type <a> exp                    /* set data type for expressions */
%type <s> NAME
%type <d> NUMBER

%nonassoc LOWER_THAN_ELSE
%nonassoc ELSE

// %token <pos_int> POS_INTEGER

%%

top:
  exp { *result = $1;};

// line: /* nothing */
//  | line exp EOL {
//     // TODO
//     //printf("= %g\n", eval($2)); //evaluate and print the AST
//     //printf("> ");
//    }
//  | line EOL { printf("> "); } /* blank line or a comment */
//  ;

exp: NUMBER                               { $$ = new RawAST(NUMBER_NODE, $1); }
  | exp ADD exp                           { $$ = new RawAST(ADDITION_NODE, {$1, $3}); }
  | exp SUB exp                           { $$ = new RawAST(SUBTRACTION_NODE, {$1, $3}); }
  | exp MUL exp                           { $$ = new RawAST(MULTIPLY_NODE, {$1, $3}); }
  | exp DIV exp                           { $$ = new RawAST(DIVIDE_NODE, {$1, $3}); }
  | OP exp CP                             { $$ = $2; }
  | O_SQ exp COLON exp C_SQ               { $$ = new RawAST(RANGE_NODE, {$2, $4}); }
  | NAME                                  { $$ = new RawAST(REFERENCE_NODE, $1);
                                            free($1);
                                          }
  | NAME EQUALS exp                       { $$ = new RawAST(REFERENCE_SET, $1, $3);
                                            free($1);
                                          }
  | NAME EQUALS exp CM exp                { $$ = new RawAST(REFERENCE_SET, $1, $3, $5);
                                            free($1);
                                          }
  | REDUCE OP exp CM NAME CP              { $$ = new RawAST(REDUCE_NODE, $3, $5);
                                            free($5);
                                          }
  | REDUCE OP exp CM NAME CM exp CP       { $$ = new RawAST(REDUCE_NODE, $3, $5, $7);
                                            free($5);
                                          }
  | SELECT OP exp CM NAME CP              { $$ = new RawAST(SELECT_NODE, $3, $5);
                                            free($5);
                                          }
  | SELECT OP exp CM NAME CM exp CP       { $$ = new RawAST(SELECT_NODE, $3, $5, $7);
                                            free($5);
                                          }
  // | NAME O_SQ POS_INTEGER C_SQ            { $$ = create_index_access_node($1, $3); }
  ;


%%

// void yyerror(char const *s)
// {
//   fprintf(stderr, "check error saurabh: %s\n", s);
// }
