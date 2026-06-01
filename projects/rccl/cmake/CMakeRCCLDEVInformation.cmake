# CMakeRCCLDEVInformation.cmake
# Compile and link rules for the RCCLDEV custom language.

# Tell CMake how to format -D, -I, -isystem flags for this language.
set(CMAKE_INCLUDE_FLAG_RCCLDEV "-I")
set(CMAKE_INCLUDE_SYSTEM_FLAG_RCCLDEV "-isystem")

# Compile rule: source.cpp -> object.o (+ side-effect .resources.json)
# CMake substitutes <DEFINES>, <INCLUDES>, <FLAGS> from the target.
set(CMAKE_RCCLDEV_COMPILE_OBJECT
  "<CMAKE_RCCLDEV_COMPILER> --compile <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> <SOURCE>")

# Link rule: objects -> shared module (device.elf)
# <LINK_FLAGS> carries target_link_options (e.g. --dispatcher=..., --arch=...).
# <OBJECTS> is the list of compiled object files.
set(CMAKE_RCCLDEV_CREATE_SHARED_MODULE
  "<CMAKE_RCCLDEV_COMPILER> --link <LINK_FLAGS> -o <TARGET> <OBJECTS>")

# Also support SHARED library creation (same rule)
set(CMAKE_RCCLDEV_CREATE_SHARED_LIBRARY
  "<CMAKE_RCCLDEV_COMPILER> --link <LINK_FLAGS> -o <TARGET> <OBJECTS>")

set(CMAKE_RCCLDEV_OUTPUT_EXTENSION ".o")
set(CMAKE_RCCLDEV_OUTPUT_EXTENSION_REPLACE 1)

# Source file extensions handled by this language
set(CMAKE_RCCLDEV_SOURCE_FILE_EXTENSIONS "rccldev.cpp")

# Linker preference (lower = preferred less for mixed-language linking)
set(CMAKE_RCCLDEV_LINKER_PREFERENCE 10)

# Inform cmake that RCCLDEV objects are ELF
set(CMAKE_RCCLDEV_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_RCCLDEV_IMPLICIT_LINK_DIRECTORIES "")
