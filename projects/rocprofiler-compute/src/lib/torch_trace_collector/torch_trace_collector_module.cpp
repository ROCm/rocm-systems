// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Python module exposing the torch_trace_collector API. Uses the CPython C
// API rather than pybind11, which only ships inside the PyTorch wheel.

#include "process_state.h"
#include "record_function_installation.h"
#include "user_scope.h"

#include <Python.h>
#include <torch_abi.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

namespace
{
using namespace torch_trace_collector::detail;

// Adds one unsigned counter to stats_dict. Returns false with the Python error
// indicator set.
bool add_counter(PyObject* stats_dict, const char* name, std::uint64_t value)
{
    PyObject* counter = PyLong_FromUnsignedLongLong(value);
    if (counter == nullptr)
    {
        return false;
    }
    const int failed = PyDict_SetItemString(stats_dict, name, counter);
    Py_DECREF(counter);
    return failed == 0;
}

PyObject* py_install(PyObject* /*self*/, PyObject* /*args*/)
{
    try
    {
        return PyLong_FromLongLong(install());
    }
    catch (const std::exception& exc)
    {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
}

PyObject* py_uninstall(PyObject* /*self*/, PyObject* /*args*/)
{
    try
    {
        uninstall();
    }
    catch (const std::exception& exc)
    {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_is_installed(PyObject* /*self*/, PyObject* /*args*/)
{
    try
    {
        return PyBool_FromLong(is_installed() ? 1 : 0);
    }
    catch (const std::exception& exc)
    {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
}

PyObject* py_push_user_scope(PyObject* /*self*/, PyObject* args, PyObject* kwargs)
{
    static const char* keywords[] = {"marker", "context", "backend", nullptr};

    const char* marker  = nullptr;
    const char* context = nullptr;
    const char* backend = "";
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "ss|s", const_cast<char**>(keywords), &marker, &context, &backend) ==
        0)
    {
        return nullptr;
    }

    try
    {
        push_user_scope(marker, context, backend);
    }
    catch (const std::exception& exc)
    {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_pop_user_scope(PyObject* /*self*/, PyObject* /*args*/)
{
    try
    {
        pop_user_scope();
    }
    catch (const std::exception& exc)
    {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_dump_stats(PyObject* /*self*/, PyObject* /*args*/)
{
    const ProcessState& state = process_state();

    PyObject* stats_dict = PyDict_New();
    if (stats_dict == nullptr)
    {
        return nullptr;
    }

    PyObject* installed = PyBool_FromLong(is_installed() ? 1 : 0);
    if (installed == nullptr || PyDict_SetItemString(stats_dict, "installed", installed) != 0)
    {
        Py_XDECREF(installed);
        Py_DECREF(stats_dict);
        return nullptr;
    }
    Py_DECREF(installed);

    const bool added =
        add_counter(stats_dict, "pushes", state.stats.pushes.load()) &&
        add_counter(stats_dict, "pops", state.stats.pops.load()) &&
        add_counter(stats_dict, "user_scope_pushes", state.stats.user_scope_pushes.load()) &&
        add_counter(stats_dict, "user_scope_pops", state.stats.user_scope_pops.load()) &&
        add_counter(stats_dict, "user_scope_inherits", state.stats.user_scope_inherits.load()) &&
        add_counter(stats_dict, "snapshots_saved", state.stats.snapshots_saved.load()) &&
        add_counter(stats_dict, "snapshots_consumed", state.stats.snapshots_consumed.load()) &&
        add_counter(stats_dict, "snapshots_dropped", state.stats.snapshots_dropped.load()) &&
        add_counter(stats_dict, "snapshots_overwritten", state.stats.snapshots_overwritten.load()) &&
        add_counter(stats_dict, "callback_errors", state.stats.callback_errors.load()) &&
        add_counter(stats_dict, "snapshots_pending", state.snapshots.pending());
    if (!added)
    {
        Py_DECREF(stats_dict);
        return nullptr;
    }
    return stats_dict;
}

// Tuple of the PyTorch versions torch_abi.h describes. The loader refuses to
// trace a workload running anything else.
PyObject* make_supported_versions()
{
    constexpr std::size_t count = sizeof(torch_abi::kSupportedTorchVersions) /
                                  sizeof(torch_abi::kSupportedTorchVersions[0]);

    PyObject* versions = PyTuple_New(static_cast<Py_ssize_t>(count));
    if (versions == nullptr)
    {
        return nullptr;
    }
    for (std::size_t i = 0; i < count; ++i)
    {
        PyObject* version = PyUnicode_FromString(torch_abi::kSupportedTorchVersions[i]);
        if (version == nullptr)
        {
            Py_DECREF(versions);
            return nullptr;
        }
        PyTuple_SET_ITEM(versions, static_cast<Py_ssize_t>(i), version);
    }
    return versions;
}

PyMethodDef methods[] = {
    {"install", py_install, METH_NOARGS, "Install the global RecordFunction callback. Idempotent."},
    {"uninstall", py_uninstall, METH_NOARGS, "Remove the registered callback."},
    {"is_installed", py_is_installed, METH_NOARGS, "Return True if the callback is installed."},
    {"push_user_scope",
     reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(py_push_user_scope)),
     METH_VARARGS | METH_KEYWORDS,
     "Push a marker frame, emit a ROCTX range, and publish the stack to ThreadLocalDebugInfo."},
    {"pop_user_scope", py_pop_user_scope, METH_NOARGS, "Pop the most recent push_user_scope frame on this thread."},
    {"dump_stats", py_dump_stats, METH_NOARGS, "Return collector counters."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "torch_trace_collector",
    "Emits ROCTX ranges around PyTorch operators through a RecordFunction callback.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit_torch_trace_collector()
{
    PyObject* module = PyModule_Create(&module_def);
    if (module == nullptr)
    {
        return nullptr;
    }

    PyObject* versions = make_supported_versions();
    if (versions == nullptr || PyModule_AddObject(module, "supported_torch_versions", versions) != 0)
    {
        Py_XDECREF(versions);
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
