# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Minimum required version of CMake
cmake_minimum_required ( VERSION 3.7 )

# Find Clang package and LLVM package
find_package(Clang REQUIRED HINTS ${CMAKE_PREFIX_PATH}/llvm PATHS /opt/rocm/llvm )
find_package(LLVM REQUIRED HINTS ${CMAKE_PREFIX_PATH}/llvm PATHS /opt/rocm/llvm )

# Set the target devices
set (TARGET_DEVS "gfx1201;gfx1250") #TODO remove gfx1201 later
set (POSTFIX "1201;1250")

# If verbose output is enabled, print paths and target devices
if(${CMAKE_VERBOSE_MAKEFILE})
	get_property(clang_path TARGET clang PROPERTY LOCATION)
	get_property(objcopy_path TARGET llvm-objcopy PROPERTY LOCATION)
	message("Using clang from: ${clang_path}")
	message("Using llvm-objcopy from: ${objcopy_path}")
	message("Preamble Shaders assembled for: ${TARGET_DEVS}")
endif()

# Function to generate kernel bitcode
function(gen_kernel_bc TARGET_ID INPUT_FILE OUTPUT_FILE)
	set(CODE_OBJECT "${OUTPUT_FILE}.hsaco")

	# Separate clang arguments
	separate_arguments(CLANG_ARG_LIST UNIX_COMMAND "-x assembler -target amdgcn-amd-amdhsa -mcode-object-version=5 -fPIC -mcpu=${TARGET_ID} -o ${CODE_OBJECT} ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE}")

	# Add custom command to generate the kernel bitcode
	add_custom_command(OUTPUT ${CODE_OBJECT} COMMAND clang ${CLANG_ARG_LIST}
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE} clang
	COMMENT "BUILDING bitcode for ${OUTPUT_FILE}..."
	VERBATIM)

	separate_arguments(OBJCOPY_ARG_LIST UNIX_COMMAND "--dump-section=.text=${OUTPUT_FILE} ${CODE_OBJECT}")

	# Add custom command to extract binary from the bitcode
	add_custom_command(OUTPUT ${OUTPUT_FILE}
	COMMAND llvm-objcopy ${OBJCOPY_ARG_LIST}
	DEPENDS ${CODE_OBJECT} llvm-objcopy
	COMMENT "Extracting binary for ${OUTPUT_FILE}..."
	VERBATIM)

	if(${CMAKE_VERBOSE_MAKEFILE})
		message("     Preamble Shader Source: " ${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_FILE})
		message("     Preamble Shader Binary: " ${OUTPUT_FILE})
	endif()

endfunction(gen_kernel_bc)

# Function to build a kernel for each target device
function(build_kernel PREAMBLE_SHADER_NAME PREAMBLE_FILE TARGET_ID)
	set(CODE_OBJECT_FILE "${PREAMBLE_SHADER_NAME}${target_id}")
	gen_kernel_bc(${TARGET_ID} ${PREAMBLE_FILE} ${CODE_OBJECT_FILE})
	list(APPEND HSACO_TARG_LIST "${CODE_OBJECT_FILE}")
	set(HSACO_TARG_LIST ${HSACO_TARG_LIST} PARENT_SCOPE)
endfunction(build_kernel)

# Function to build kernels for all devices and shaders
function(build_kernels_for_devices SHADER_NAMES SHADER_FILES)
	set(HSACO_TARG_LIST "")

	list(LENGTH TARGET_DEVS num_target_devices)
	math(EXPR num_target_devices "${num_target_devices} - 1")
	list(LENGTH SHADER_NAMES num_shader_names)
	math(EXPR num_shader_names "${num_shader_names} - 1")

	foreach(shader_index RANGE ${num_shader_names})
		list(GET SHADER_NAMES ${shader_index} shader_name)
		list(GET SHADER_FILES ${shader_index} shader_file)
		foreach(device_index RANGE ${num_target_devices})
			# Get device from list of target devices
			list(GET TARGET_DEVS ${device_index} target_device)
			# Get postfix from list of postfixes
			list(GET POSTFIX ${device_index} postfix)
			if(${CMAKE_VERBOSE_MAKEFILE})
				message("\n  Generating: ${target_device} for ${shader_name} ...")
			endif()

			# Define the name of the code object file
			set(CODE_OBJECT_FILE "${shader_name}${postfix}")

			# Generate the kernel bitcode for the current device and shader
			gen_kernel_bc(${target_device} ${shader_file} ${CODE_OBJECT_FILE})
			# Append the code object file to the list
			list(APPEND HSACO_TARG_LIST "${CODE_OBJECT_FILE}")
		endforeach(device_index)
	endforeach(shader_index)

	# Make the list of code object files available in the parent scope
	set(HSACO_TARG_LIST ${HSACO_TARG_LIST} PARENT_SCOPE)

endfunction(build_kernels_for_devices)


# Function to generate the bytecode stream and create the header file
function(generate_bytecodeStrm HeaderFILE)
	set(ARG_LIST "${CMAKE_CURRENT_BINARY_DIR}/${HeaderFILE}.h")
	# Copy the shell script to the build directory
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/create_preamble_shader_header.sh
		${CMAKE_CURRENT_BINARY_DIR}/create_preamble_shader_header.sh
		COPYONLY)

	# Add a custom command to generate the header file
if (UNIX)
	add_custom_command(OUTPUT ${HeaderFILE}.h
		COMMAND ${CMAKE_CURRENT_BINARY_DIR}/create_preamble_shader_header.sh ${ARG_LIST} ${HSACO_TARG_LIST}
		COMMENT "Collating preamble shaders..."
		DEPENDS ${HSACO_TARG_LIST} ${CMAKE_CURRENT_BINARY_DIR}/create_preamble_shader_header.sh)
else()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_custom_command(
    OUTPUT ${HeaderFILE}.h
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/create_preamble_shader_header.py ${ARG_LIST} ${HSACO_TARG_LIST}
    COMMENT "Collating preamble shaders..."
    DEPENDS ${HSACO_TARG_LIST} create_preamble_shader_header.py)
endif()
	# Add a custom target that depends on the header file
	add_custom_target(${HeaderFILE} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${HeaderFILE}.h)

endfunction(generate_bytecodeStrm)

# Build kernels for device code preamble (one call so HSACO_TARG_LIST is complete)
build_kernels_for_devices("kCodePreamble" "preamble_shader.s")
# Generate bytecode stream
generate_bytecodeStrm("amd_preamble_shaders")
