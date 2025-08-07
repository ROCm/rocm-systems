/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_YY_ROCPROFILER_SOURCE_LIB_ROCPROFILER_SDK_COUNTERS_PARSER_PARSER_H_INCLUDED
#define YY_YY_ROCPROFILER_SOURCE_LIB_ROCPROFILER_SDK_COUNTERS_PARSER_PARSER_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
#    define YYDEBUG 1
#endif
#if YYDEBUG
extern int yydebug;
#endif
/* "%code requires" blocks.  */
#line 2 "parser.y"

#include "raw_ast.hpp"
using namespace rocprofiler::counters;
#define YYDEBUG 1

#line 55 "parser.h"

/* Token kinds.  */
#ifndef YYTOKENTYPE
#    define YYTOKENTYPE
enum yytokentype
{
    YYEMPTY         = -2,
    YYEOF           = 0,   /* "end of file"  */
    YYerror         = 256, /* error  */
    YYUNDEF         = 257, /* "invalid token"  */
    ADD             = 258, /* ADD  */
    SUB             = 259, /* SUB  */
    MUL             = 260, /* MUL  */
    DIV             = 261, /* DIV  */
    ABS             = 262, /* ABS  */
    EQUALS          = 263, /* EQUALS  */
    OP              = 264, /* OP  */
    CP              = 265, /* CP  */
    O_SQ            = 266, /* O_SQ  */
    C_SQ            = 267, /* C_SQ  */
    COLON           = 268, /* COLON  */
    EOL             = 269, /* EOL  */
    UMINUS          = 270, /* UMINUS  */
    CM              = 271, /* CM  */
    NUMBER          = 272, /* NUMBER  */
    RANGE           = 273, /* RANGE  */
    NAME            = 274, /* NAME  */
    REDUCE          = 275, /* REDUCE  */
    SELECT          = 276, /* SELECT  */
    ACCUMULATE      = 277, /* ACCUMULATE  */
    DIM_ARGS_RANGE  = 278, /* DIM_ARGS_RANGE  */
    LOWER_THAN_ELSE = 279, /* LOWER_THAN_ELSE  */
    ELSE            = 280  /* ELSE  */
};
typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if !defined YYSTYPE && !defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#    line 34 "parser.y"

    RawAST*     a;  /* For ast node */
    LinkedList* ll; /* For linked list node */
    int64_t     d;
    char*       s;

#    line 103 "parser.h"
};
typedef union YYSTYPE YYSTYPE;
#    define YYSTYPE_IS_TRIVIAL  1
#    define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;

int
yyparse(RawAST** result);

#endif /* !YY_YY_ROCPROFILER_SOURCE_LIB_ROCPROFILER_SDK_COUNTERS_PARSER_PARSER_H_INCLUDED  */
