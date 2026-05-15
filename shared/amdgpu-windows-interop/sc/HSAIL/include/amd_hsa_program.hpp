//-----------------------------------------------------------------------------
// Copyright (c) 2011 - 2015 Advanced Micro Devices, Inc.  All rights reserved.
//-----------------------------------------------------------------------------

/// @file    amd_hsa_program.hpp
/// @author  AMD HSA Finalizer Team
///
/// @brief   Public AMD HSA Program Interfaces.
#ifndef AMD_HSA_PROGRAM_HPP
#define AMD_HSA_PROGRAM_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "amd_hsa_code.hpp"
#include "Brig.h"
#include "hsa.h"
#include "hsa_ext_finalize.h"

/// @brief Descriptive version of AMD HSA Program.
#define AMD_HSA_PROGRAM_VERSION "AMD HSA Program v1.04 (August 3, 2015)"
#define AMD_HSA_PROGRAM_VERSION_MAJOR 1
#define AMD_HSA_PROGRAM_VERSION_MINOR 4

/// @brief Environment variable. If set, overrides options parameter from
/// Program::Create with contents of this environment variable.
#define ENVVAR_AMD_HSA_PROGRAM_CREATE_OPTIONS "AMD_HSA_PROGRAM_CREATE_OPTIONS"

/// @brief Environment variable. If set, concatenates options parameter from
/// Program::Create with contents of this environment variable.
#define ENVVAR_AMD_HSA_PROGRAM_CREATE_OPTIONS_APPEND "AMD_HSA_PROGRAM_CREATE_OPTIONS_APPEND"

/// @brief Environment variable. If set, overrides options parameter from
/// Program::Finalize with contents of this environment variable.
#define ENVVAR_AMD_HSA_PROGRAM_FINALIZE_OPTIONS "AMD_HSA_PROGRAM_FINALIZE_OPTIONS"

/// @brief Environment variable. If set, concatenates options parameter from
/// Program::Finalize with contents of this environment variable.
#define ENVVAR_AMD_HSA_PROGRAM_FINALIZE_OPTIONS_APPEND "AMD_HSA_PROGRAM_FINALIZE_OPTIONS_APPEND"

/// @brief AMD HSA Program attributes (in addition to hsa_ext_program_info_t,
/// which is defined in HSA Runtime Specification), enumeration values below
/// must be negative.
typedef int32_t amd_hsa_program_info32_t;
enum amd_hsa_program_info_t {
  /// @brief Major version of BRIG specified when AMD HSA Program was created.
  /// The type of this attribute is BrigVersion32_t.
  AMD_HSA_PROGRAM_INFO_BRIG_VERSION_MAJOR = -1,
  /// @brief Minor version of BRIG specified when AMD HSA Program was created.
  /// The type of this attribute is BrigVersion32_t.
  AMD_HSA_PROGRAM_INFO_BRIG_VERSION_MINOR = -2,
  /// @brief Indicates whether or not AMD HSA Program was created with debugging
  /// enabled. The type of this attribute is bool.
  AMD_HSA_PROGRAM_INFO_IS_DEBUGGING_ENABLED = -3
};

namespace amd {
namespace hsa {
namespace program {

/// @class Context
class Context {
public:
  /// @brief Default destructor.
  virtual ~Context() {}

  /// @brief Invoked when AMD HSA Program needs to allocate @p size bytes of
  /// code object memory whose alignment is specified by @p align.
  ///
  /// @param[in] size Requested allocation size in bytes.
  /// @param[in] align Requested alignment.
  ///
  /// @returns Pointer to allocated code object memory on success, null pointer
  /// on failure.
  virtual void* CodeObjectAlloc(size_t size, size_t align) = 0;

  /// @brief Invoked when AMD HSA Program needs to copy @p size bytes from
  /// memory pointed to by @p src to code object memory pointed to by @p dst.
  ///
  /// @param[in] dst Pointer to code object memory to copy to.
  /// @param[in] src Pointer to memory to copy from.
  /// @param[in] size Requested copy size in bytes.
  ///
  /// @returns True on success, false on failure.
  virtual bool CodeObjectCopy(void *dst, const void *src, size_t size) = 0;

  /// @brief Invoked when AMD HSA Program needs to deallocate @p size bytes of
  /// code object memory pointed to by @p ptr.
  ///
  /// @param[in] ptr Pointer to code object memory to deallocate.
  /// @param[in] size Requested deallocation size in bytes.
  virtual void CodeObjectFree(void *ptr, size_t size) = 0;

  /// @brief Invoked when AMD HSA Finalizer and Program needs to reprot message or error
  ///
  /// @param[in] str Message to report.
  virtual void ReportMessage(const std::string& str) = 0;

protected:
  /// @brief Default constructor.
  Context() {}

private:
  /// @brief Copy constructor - not available.
  Context(const Context&);

  /// @brief Assignment operator - not available.
  Context& operator=(const Context&);
};

class Finalizer;

/// @class Program
class Program: public amd::hsa::common::Signed<0x71BB0A093D69DA92> {
public:
  /// @brief Constant BRIG Module iterator.
  typedef std::vector<hsa_ext_module_t>::const_iterator const_module_iterator;

  /// @brief BRIG Module iterator.
  typedef std::vector<hsa_ext_module_t>::iterator module_iterator;

  /// @brief Invalid HSA Program Handle.
  static const uint64_t INVALID_HANDLE = 0;

  /// @brief Destructor.
  virtual ~Program() {}

  /// @brief Converts AMD HSA Program @p program_object to HSA Program Handle.
  ///
  /// @param[in] program_object AMD HSA Program to convert. Can be null.
  ///
  /// @returns HSA Program Handle on success, invalid handle on failure.
  static hsa_ext_program_t Handle(Program *program_object);

  /// @brief Converts HSA Program Handle @p program_handle to AMD HSA Program.
  ///
  /// @param[in] program_handle HSA Program Handle to convert. Can be invalid.
  ///
  /// @returns AMD HSA Program on success, null on failure.
  static Program* Object(hsa_ext_program_t program_handle);

  /// @returns Constant iterator to first BRIG Module in AMD HSA Program. If
  /// AMD HSA Program does not contain any BRIG Modules, returned constant
  /// iterator will be equal to Program::module_end().
  virtual const_module_iterator module_begin() const = 0;

  /// @returns Constant iterator to entity following last BRIG Module in AMD
  /// HSA Program.
  virtual const_module_iterator module_end() const = 0;

  /// @returns Iterator to first BRIG Module in AMD HSA Program. If
  /// AMD HSA Program does not contain any BRIG Modules, returned
  /// iterator will be equal to Program::module_end().
  virtual module_iterator module_begin() = 0;

  /// @returns Iterator to entity following last BRIG Module in AMD
  /// HSA Program.
  virtual module_iterator module_end() = 0;

  /// @returns Context associated with AMD HSA Program.
  virtual Context* GetContext() const = 0;

  /// @returns Finalizer associated with this AMD HSA Program.
  virtual Finalizer* GetFinalizer() const = 0;

  /// @brief Retrieves current value of specified AMD HSA Program's
  /// @p attribute.
  ///
  /// @param[in] attribute AMD HSA Program's attribute to retrieve. Can be
  /// invalid.
  /// @param[out] value Pointer to client-allocated memory to store attribute's
  /// value in. Must not be null. If client-allocated memory is not large enough
  /// to hold attribute's value, behaviour is undefined.
  ///
  /// @retval HSA_STATUS_SUCCESS Function executed successfully.
  /// @retval HSA_STATUS_ERROR_INVALID_ARGUMENT Specified @p attribute is
  /// invalid AMD HSA Program's attribute.
  ///
  /// @note If function failed to execute successfully, details of failure
  /// can be retrieved using Program::GetLog.
  virtual hsa_status_t GetInfo(amd_hsa_program_info32_t attribute, void *value) const = 0;

  /// @brief Adds specified BRIG Module @p module to AMD HSA Program.
  ///
  /// @details AMD HSA Program does not perform deep copy of BRIG Module
  /// upon addition, it stores pointer to BRIG Module. BRIG Module is owned by
  /// the client, which has to ensure that the lifetime of BRIG Module is
  /// greater than the lifetime of AMD HSA Program.
  ///
  /// @param[in] module BRIG Module to add. Must not be null.
  ///
  /// @retval HSA_STATUS_SUCCESS Function executed successfully.
  /// @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES Function failed to allocate
  /// resources.
  /// @retval HSA_EXT_STATUS_ERROR_INVALID_MODULE Specified @p module is invalid
  /// BRIG module.
  /// @retval HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED Specified @p module
  /// is already included in AMD HSA Program.
  /// @retval HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE Specified @p module is
  /// incompatible with AMD HSA Program.
  /// @retval HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH Symbol in specified @p module
  /// is incompatible with symbol in AMD HSA Program.
  ///
  /// @note If function failed to execute successfully, details of failure
  /// can be retrieved using Program::GetLog.
  virtual hsa_status_t AddModule(hsa_ext_module_t module) = 0;

  /// @brief Finalizes AMD HSA Program with specified @p target,
  /// @p call_convention, @p options, @p control_directives, and
  /// @p code_object_type.
  ///
  /// @details Finalizes all kernels and indirect functions that belong to
  /// AMD HSA Program for specified @p target, @p call_convention,
  /// @p options, @p control_directives, and @p code_object_type. Transitive
  /// closure of all functions specified by call or scall must be defined.
  /// All kernels and indirect functions that belong to AMD HSA Program must
  /// be defined. Kernels and indirect functions that are referenced in kernels
  /// and indirect functions that belong to AMD HSA Program may or may not be
  /// defined, but must be declared. All global and readonly variables that
  /// belong to AMD HSA Program, or referenced in kernels and indirect functions
  /// that belong to AMD HSA Program may or may not be defined, but must be
  /// declared.
  ///
  /// @param[in] target Target to finalize for. Must not be null.
  /// @param[in] call_convention Call convention to finalize for. Must be valid.
  /// @param[in] options Options to finalize for. Can be null.
  /// @param[in] control_directives Control directives to finalize for. Can be
  /// invalid.
  /// @param[in] code_object_type Code object type to create. Must be valid.
  /// @param[out] code_object Code object generated by AMD HSA Program. Must
  /// not be null.
  ///
  /// @retval HSA_STATUS_SUCCESS Function executed successfully.
  /// @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES Function failed to allocate
  /// resources.
  /// @retval HSA_STATUS_ERROR_INVALID_ISA Specified @p target is invalid.
  /// @retval HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH Specified
  /// @p control_directives does not match control directives in
  /// one of kernels or indirect functions that belong to AMD HSA Program.
  /// @retval HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED AMD HSA Program failed
  /// to finalize.
  ///
  /// @note If function failed to execute successfully, details of failure
  /// can be retrieved using Program::GetLog.
  ///
  /// @deprecated @p control_directives will be included in @p options starting
  /// AMD HSA Program v2.0.
  virtual hsa_status_t Finalize(
    const char *target,
    int32_t call_convention,
    const char *options,
    hsa_ext_control_directives_t control_directives,
    hsa_code_object_type_t code_object_type,
    hsa_code_object_t *code_object) = 0;

protected:
  /// @brief Default constructor.
  Program() {}

private:
  /// @brief Copy constructor - not available.
  Program(const Program&);

  /// @brief Assignment operator - not available.
  Program& operator=(const Program&);
};


/// @class Finalizer
class Finalizer {
public:
  /// @brief Destructor.
  virtual ~Finalizer() {}

  /// @brief Creates AMD HSA Finalizer with specified @p context.
  ///
  /// @param[in] context Context. Must not be null.
  ///
  /// @returns AMD HSA Finalizer on success, null on failure.
  static Finalizer* CreateFinalizer(Context* context);

  /// @brief Destroys AMD HSA Finalizer @p finalizer_object.
  ///
  /// @param[in] finalizer_object AMD HSA Finalizer to destroy. Must not be null.
  static void DestroyFinalizer(Finalizer *finalizer_object);

  /// @brief Creates empty AMD HSA Program with specified @p profile,
  /// @p machine_model, @p rounding_mode, @p options, @p context, @p major and
  /// @p minor BRIG versions.
  ///
  /// @param[in] profile HSA profile. Must be valid.
  /// @param[in] machine_model HSA machine model. Must be valid.
  /// @param[in] rounding_mode HSA rounding mode. Must be valid.
  /// @param[in] options User options. Can be null.
  /// @param[in] brig_major Major BRIG version. Must be valid.
  /// @param[in] brig_minor Minor BRIG version. Must be valid.
  ///
  /// @returns AMD HSA Program on success, null on failure.
  virtual Program* CreateProgram(
    hsa_profile_t profile,
    hsa_machine_model_t machine_model,
    hsa_default_float_rounding_mode_t rounding_mode,
    const char *options,
    BrigVersion32_t brig_major = BRIG_VERSION_BRIG_MAJOR,
    BrigVersion32_t brig_minor = BRIG_VERSION_BRIG_MINOR) = 0;

  /// @brief Destroys AMD HSA Program @p program_object.
  ///
  /// @param[in] program_object AMD HSA Program to destroy. Must not be null.
  virtual void DestroyProgram(Program *program_object) = 0;

  /// @brief Prints available finalizer options as error and exits.
  virtual void PrintFinalizerOptions() const = 0;

  /// @returns Context associated with Finalizer.
  virtual Context* GetContext() const = 0;

  /// @brief Enables code cache optimization.
  virtual void EnableCodeCache() = 0;

  /// @brief Disables code cache optimization.
  virtual void DisableCodeCache() = 0;

  /// @returns True if code cache is enabled, false otherwise.
  virtual bool IsCodeCacheEnabled() const = 0;

  /// @returns List of names for supported targets.
  virtual const std::vector<std::string>& GetSupportedTargets() const = 0;

protected:
  /// @brief Default constructor.
  Finalizer() {}

private:
  /// @brief Copy constructor - not available.
  Finalizer(const Finalizer&);

  /// @brief Assignment operator - not available.
  Finalizer& operator=(const Finalizer&);
};

} // namespace program
} // namespace hsa
} // namespace amd

#endif // AMD_HSA_PROGRAM_HPP
