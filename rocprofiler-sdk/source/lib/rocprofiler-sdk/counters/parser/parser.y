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

#include "raw_ast.hpp"

int yyparse(rocprofiler::counters::RawAST** result);
int yylex(void);
void yyerror(rocprofiler::counters::RawAST**, const char *s) { ROCP_ERROR << s; }
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
 LinkedList* ll;     /* For linked list node */
 int64_t d;
 char* s;
}

%token NUMBER RANGE              /* set data type for numbers */
%token NAME                      /* set data type for variables and user-defined functions */
%token REDUCE SELECT             /* set data type for special functions */
%token ACCUMULATE
%token DIM_ARGS_RANGE
%type <a> exp                    /* set data type for expressions */
%type <s> NAME DIM_ARGS_RANGE
%type <d> NUMBER
%type <ll> reduce_dim_args select_dim_args

%nonassoc LOWER_THAN_ELSE
%nonassoc ELSE

// %token <pos_int> POS_INTEGER

%%

top:
  exp { *result = $1;};


exp: NUMBER                               { $$ = new RawAST(NUMBER_NODE, $1); }
  | exp ADD exp                           { $$ = new RawAST(ADDITION_NODE, {$1, $3}); }
  | exp SUB exp                           { $$ = new RawAST(SUBTRACTION_NODE, {$1, $3}); }
  | exp MUL exp                           { $$ = new RawAST(MULTIPLY_NODE, {$1, $3}); }
  | exp DIV exp                           { $$ = new RawAST(DIVIDE_NODE, {$1, $3}); }
  | OP exp CP                             { $$ = $2; }
  | NAME                                  { $$ = new RawAST(REFERENCE_NODE, $1);
                                            free($1);
                                          }
  | ACCUMULATE OP NAME CM NAME CP          {
                                            $$ = new RawAST(ACCUMULATE_NODE, $3, $5);
                                            free($3);
                                            free($5);
                                          }
  | REDUCE OP exp CM NAME CP              {
                                            $$ = new RawAST(REDUCE_NODE, $3, $5, NULL);
                                            free($5);
                                          }
  | REDUCE OP exp CM NAME CM O_SQ reduce_dim_args C_SQ CP {
                                            $$ = new RawAST(REDUCE_NODE, $3, $5, $8);
                                            free($5);
                                          }
  | SELECT OP exp CM O_SQ select_dim_args C_SQ CP {
                                            $$ = new RawAST(SELECT_NODE, $3, $6);
                                          }
  ;



reduce_dim_args: NAME                     { $$ = new LinkedList($1, NULL);
                                            free($1);
                                          }
 | NAME CM reduce_dim_args               { $$ = new LinkedList($1, $3);
                                            free($1);
                                          }
 ;



select_dim_args: NAME EQUALS O_SQ NUMBER C_SQ { 
                                            $$ = new LinkedList($1, $4, NULL);
                                            free($1);
                                          }
 | NAME EQUALS O_SQ NUMBER C_SQ CM select_dim_args { 
                                            $$ = new LinkedList($1, $4, $7);
                                            free($1);
                                          }
 | NAME EQUALS O_SQ DIM_ARGS_RANGE C_SQ { 
                                            $$ = new LinkedList($1, $4, NULL);
                                            free($1);
                                            free($4);
                                          }
 | NAME EQUALS O_SQ DIM_ARGS_RANGE C_SQ CM select_dim_args { 
                                            $$ = new LinkedList($1, $4, $7);
                                            free($1);
                                            free($4);
                                          }
 ;


%%
