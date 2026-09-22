// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/common/defines.hpp"

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstdint>

extern "C" {
struct pysqlite_state;
struct sqlite3;

enum autocommit_mode
{
    AUTOCOMMIT_LEGACY   = -1,
    AUTOCOMMIT_ENABLED  = 1,
    AUTOCOMMIT_DISABLED = 0,
};

typedef struct _callback_context
{
    PyObject*       callable;
    PyObject*       module;
    pysqlite_state* state;
} callback_context;

// this is the python sqlite3 wrapper from Python 3.12.2
typedef struct
{
    PyObject_HEAD;
    sqlite3*        db;
    pysqlite_state* state;

    /* the type detection mode. Only 0, PARSE_DECLTYPES, PARSE_COLNAMES or a
     * bitwise combination thereof makes sense */
    int detect_types;

    /* NULL for autocommit, otherwise a string with the isolation level */
    const char*          isolation_level;
    enum autocommit_mode autocommit;

    /* 1 if a check should be performed for each API call if the connection is
     * used from the same thread it was created in */
    int check_same_thread;

    int initialized;

    /* thread identification of the thread the connection was created in */
    unsigned long thread_ident;

    PyObject* statement_cache;

    /* Lists of weak references to cursors and blobs used within this connection */
    PyObject* cursors;
    PyObject* blobs;

    /* Counters for how many cursors were created in the connection. May be
     * reset to 0 at certain intervals */
    int created_cursors;

    PyObject* row_factory;

    /* Determines how bytestrings from SQLite are converted to Python objects:
     * - PyUnicode_Type:        Python Unicode objects are constructed from UTF-8 bytestrings
     * - PyBytes_Type:          The bytestrings are returned as-is.
     * - Any custom callable:   Any object returned from the callable called with the bytestring
     *                          as single parameter.
     */
    PyObject* text_factory;

    // Remember contexts used by the trace, progress, and authoriser callbacks
    callback_context* trace_ctx;
    callback_context* progress_ctx;
    callback_context* authorizer_ctx;

    /* Exception objects: borrowed refs. */
    PyObject* Warning;
    PyObject* Error;
    PyObject* InterfaceError;
    PyObject* DatabaseError;
    PyObject* DataError;
    PyObject* OperationalError;
    PyObject* IntegrityError;
    PyObject* InternalError;
    PyObject* ProgrammingError;
    PyObject* NotSupportedError;
} pysqlite_Connection;
}
