/*
 * MIT License
 *
 * Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * test_roofline.cpp
 *
 * C++ program that embeds Python interpreter to call test_roofline.py main() function
 * and converts the returned roofline data to C++ native data structures.
 *
 * Compilation:
 *   g++ -o test_roofline test_roofline.cpp \
 *       -I/usr/include/python3.10 \
 *       -lpython3.10 \
 *       -std=c++17
 *
 * Usage:
 *   ./test_roofline
 */

#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iomanip>

// Structure to hold roofline data for a single kernel
struct RooflineKernelData {
    std::string kernel_name;
    double total_flops;
    double l1_cache_data;
    double l2_cache_data;
    double hbm_cache_data;
};

// Structure to hold roofline data for a workload
struct RooflineWorkloadData {
    std::string workload_path;
    std::vector<RooflineKernelData> kernels;
};

// Result type
using RooflineResult = std::map<std::string, std::vector<RooflineKernelData>>;

// Helper function to extract string from Python object
std::string PyObjectToString(PyObject* obj) {
    if (!obj || obj == Py_None) {
        return "";
    }
    PyObject* str_obj = PyObject_Str(obj);
    if (!str_obj) {
        return "";
    }
    const char* c_str = PyUnicode_AsUTF8(str_obj);
    std::string result = c_str ? c_str : "";
    Py_DECREF(str_obj);
    return result;
}

// Helper function to extract double from Python object
double PyObjectToDouble(PyObject* obj) {
    if (!obj || obj == Py_None) {
        return 0.0;
    }
    return PyFloat_AsDouble(obj);
}

// Convert pandas DataFrame to vector of RooflineKernelData
std::vector<RooflineKernelData> ConvertDataFrame(PyObject* df) {
    std::vector<RooflineKernelData> kernels;

    if (!df || df == Py_None) {
        std::cerr << "DataFrame is None" << std::endl;
        return kernels;
    }

    // Get the 'to_dict' method to convert DataFrame to dictionary
    PyObject* to_dict_method = PyObject_GetAttrString(df, "to_dict");
    if (!to_dict_method) {
        std::cerr << "Failed to get to_dict method" << std::endl;
        PyErr_Print();
        return kernels;
    }

    // Call to_dict('records') to get list of dictionaries
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString("records"));
    PyObject* records = PyObject_CallObject(to_dict_method, args);
    Py_DECREF(args);
    Py_DECREF(to_dict_method);

    if (!records) {
        std::cerr << "Failed to convert DataFrame to records" << std::endl;
        PyErr_Print();
        return kernels;
    }

    // Iterate over records (list of dictionaries)
    Py_ssize_t num_records = PyList_Size(records);
    for (Py_ssize_t i = 0; i < num_records; i++) {
        PyObject* record = PyList_GetItem(records, i);  // Borrowed reference
        if (!record) continue;

        RooflineKernelData kernel;

        // Extract fields from dictionary
        PyObject* kernel_name_obj = PyDict_GetItemString(record, "kernel_name");
        kernel.kernel_name = PyObjectToString(kernel_name_obj);

        PyObject* total_flops_obj = PyDict_GetItemString(record, "total_flops");
        kernel.total_flops = PyObjectToDouble(total_flops_obj);

        PyObject* l1_cache_obj = PyDict_GetItemString(record, "l1_cache_data");
        kernel.l1_cache_data = PyObjectToDouble(l1_cache_obj);

        PyObject* l2_cache_obj = PyDict_GetItemString(record, "l2_cache_data");
        kernel.l2_cache_data = PyObjectToDouble(l2_cache_obj);

        PyObject* hbm_cache_obj = PyDict_GetItemString(record, "hbm_cache_data");
        kernel.hbm_cache_data = PyObjectToDouble(hbm_cache_obj);

        kernels.push_back(kernel);
    }

    Py_DECREF(records);
    return kernels;
}

// Main function to call Python and convert results
RooflineResult CallPythonRooflineTest() {
    RooflineResult result;

    // Initialize Python interpreter
    Py_Initialize();

    // Add both possible paths to Python path to support running from:
    // 1. Root directory: ./test_roofline/test_roofline_cpp
    // 2. test_roofline directory: ./test_roofline_cpp
    PyObject* sys_path = PySys_GetObject("path");  // Borrowed reference
    if (sys_path) {
        // Add current directory (for running from test_roofline/)
        PyObject* current_dir = PyUnicode_FromString(".");
        PyList_Append(sys_path, current_dir);
        Py_DECREF(current_dir);

        // Add test_roofline directory (for running from root)
        PyObject* test_dir = PyUnicode_FromString("./test_roofline");
        PyList_Append(sys_path, test_dir);
        Py_DECREF(test_dir);
    }

    // Import test_roofline module
    PyObject* module_name = PyUnicode_FromString("test_roofline");
    PyObject* module = PyImport_Import(module_name);
    Py_DECREF(module_name);

    if (!module) {
        std::cerr << "Failed to import test_roofline module" << std::endl;
        PyErr_Print();
        Py_Finalize();
        return result;
    }

    // Get main function
    PyObject* main_func = PyObject_GetAttrString(module, "main");
    if (!main_func || !PyCallable_Check(main_func)) {
        std::cerr << "Cannot find or call main() function" << std::endl;
        Py_XDECREF(main_func);
        Py_DECREF(module);
        Py_Finalize();
        return result;
    }

    // Call main function
    std::cout << "Calling Python main() function..." << std::endl;
    PyObject* py_result = PyObject_CallObject(main_func, nullptr);
    Py_DECREF(main_func);
    Py_DECREF(module);

    if (!py_result) {
        std::cerr << "Failed to call main() function" << std::endl;
        PyErr_Print();
        Py_Finalize();
        return result;
    }

    // Check if result is None (error case)
    if (py_result == Py_None) {
        std::cerr << "main() returned None (error occurred)" << std::endl;
        Py_DECREF(py_result);
        Py_Finalize();
        return result;
    }

    // Result should be a dictionary mapping workload paths to DataFrames
    if (!PyDict_Check(py_result)) {
        std::cerr << "Result is not a dictionary" << std::endl;
        Py_DECREF(py_result);
        Py_Finalize();
        return result;
    }

    // Iterate over dictionary items
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(py_result, &pos, &key, &value)) {
        std::string workload_path = PyObjectToString(key);
        std::vector<RooflineKernelData> kernels = ConvertDataFrame(value);
        result[workload_path] = kernels;
    }

    Py_DECREF(py_result);

    // Finalize Python interpreter
    Py_Finalize();

    return result;
}

// Print roofline results
void PrintRooflineResults(const RooflineResult& results) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "C++ Roofline Results" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    for (const auto& [workload_path, kernels] : results) {
        std::cout << "\nWorkload: " << workload_path << std::endl;
        std::cout << "Number of kernels: " << kernels.size() << std::endl;
        std::cout << "\n";

        // Print table header
        std::cout << std::left
                  << std::setw(20) << "Kernel Name"
                  << std::setw(15) << "Total FLOPs"
                  << std::setw(15) << "AI L1"
                  << std::setw(15) << "AI L2"
                  << std::setw(15) << "AI HBM"
                  << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        // Print kernel data
        for (const auto& kernel : kernels) {
            std::cout << std::left
                      << std::setw(20) << kernel.kernel_name
                      << std::setw(15) << std::fixed << std::setprecision(1) << kernel.total_flops
                      << std::setw(15) << std::fixed << std::setprecision(1) << kernel.l1_cache_data
                      << std::setw(15) << std::fixed << std::setprecision(1) << kernel.l2_cache_data
                      << std::setw(15) << std::fixed << std::setprecision(1) << kernel.hbm_cache_data
                      << std::endl;
        }
    }
}

int main() {
    std::cout << "Starting C++ test for roofline data..." << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Call Python test and get results
    RooflineResult results = CallPythonRooflineTest();

    // Check if we got results
    if (results.empty()) {
        std::cerr << "\nERROR: No roofline data returned" << std::endl;
        return 1;
    }

    // Print results
    PrintRooflineResults(results);

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "SUCCESS: C++ program completed successfully" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
