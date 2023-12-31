# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2019 CAMELab
#
# Author: Donghyun Gouk <kukdh1@camelab.org>
#

# Set version and project name
cmake_minimum_required(VERSION 3.10)
project(simplessd)

# Validate build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Select CMake build type." FORCE)

  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

if(CMAKE_BUILD_TYPE AND NOT uppercase_CMAKE_BUILD_TYPE MATCHES "^(DEBUG|RELEASE)$")
  message(FATAL_ERROR "Invalid value for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

# Validate build path
# https://github.com/llvm/llvm-project/blob/master/llvm/CMakeLists.txt#L249
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR AND NOT MSVC)
  message(FATAL_ERROR "In-source builds are not allowed.
Please create a directory and run cmake from there, passing the path
to this source directory as the last argument.
This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
Please delete them.")
endif()

# Set cache entry for external library include path
set(DRAMPOWER_SOURCE_DIR "." CACHE PATH "Source directory to DRAMPower library")

# Add options for firmware latency generation
option(FW_LATENCY "Use automatic firmware latency generation using LLVM." OFF)

set(FW_CROSS_COMPILE "" CACHE STRING "Set cross-compiler prefix. E.g.: -DFW_CROSS_COMPILE=arm-none-eabi-")
set(FW_TARGET "" CACHE STRING "Set compile target (clang++ --target=<FW_TARGET>). E.g.: arm-eabi")
set(FW_CPU "" CACHE STRING "Set CPU to use (clang++ -mcpu=<FW_CPU>). E.g.: cortex-r52")

# Add options for sanitizer
option(USE_SANITIZER "Use sanitizer." OFF)

# Set output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Check DRAMPOWER_SOURCE_DIR is valid
if(NOT EXISTS "${DRAMPOWER_SOURCE_DIR}/libdrampower/LibDRAMPower.h")
  message(FATAL_ERROR "Failed to find DRAMPower library. \
Check you specified DRAMPower path by -DDRAMPOWER_SOURCE_DIR.")
endif()

# Check mcpat is valid
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/lib/mcpat/mcpat.h")
  message(FATAL_ERROR "Failed to find McPAT library. \
Check you initialized submodules.")
endif()

# Check pugixml is valid
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/lib/pugixml/src/pugixml.hpp")
  message(FATAL_ERROR "Failed to find pugixml library. \
Check you initialized submodules.")
endif()

# Check cache2 is valid
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/lib/catch2/include/catch.hpp")
  message(FATAL_ERROR "Failed to find catch2 library. \
Check you initialized submodules.")
endif()

# Check json is valid
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/lib/json/include/nlohmann/json.hpp")
  message(FATAL_ERROR "Failed to find nlohmann json c++ library. \
Check you initialized submodules.")
endif()

# Add subproject
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/mcpat)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/pugixml)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/catch2)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/json)

# Set include directories
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/lib/mcpat
  ${PROJECT_SOURCE_DIR}/lib/mcpat/cacti
  ${PROJECT_SOURCE_DIR}/lib/pugixml/src
  ${PROJECT_SOURCE_DIR}/lib/catch2/single_include
  ${DRAMPOWER_SOURCE_DIR}
)

# Make version
set(INPUT_VERSION_FILE "${PROJECT_SOURCE_DIR}/sim/version.cc.in")
set(OUTPUT_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.cc")
set(PROJECT_DIRECTORY "${PROJECT_SOURCE_DIR}")
set(VERSION_TARGET "simplessd_version")

include(${PROJECT_SOURCE_DIR}/util/scripts/simplessd_version.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Specify source files
set(SRC_CONFIG
  cpu/config.cc
  mem/config.cc
  hil/config.cc
  icl/config.cc
  ftl/config.cc
  fil/config.cc
)
set(SRC_CPU
  cpu/cpu.cc
)
set(SRC_MEM
  mem/system.cc
  mem/dram/abstract_dram.cc
  mem/dram/ideal/ideal.cc
  mem/dram/simple/simple.cc
  mem/sram/abstract_sram.cc
  mem/sram/sram.cc
)
set(SRC_HIL
  hil/convert.cc
  hil/hil.cc
  hil/request.cc
)
set(SRC_HIL_COMMON
  hil/common/dma_engine.cc
  hil/common/interrupt_manager.cc
)
set(SRC_HIL_NONE
  hil/none/controller.cc
  hil/none/subsystem.cc
)
set(SRC_HIL_NVME
  hil/nvme/abstract_namespace.cc
  hil/nvme/controller.cc
  hil/nvme/namespace.cc
  hil/nvme/queue_arbitrator.cc
  hil/nvme/queue.cc
  hil/nvme/subsystem.cc
)
set(SRC_HIL_NVME_COMMAND
  hil/nvme/command/abstract_command.cc
  hil/nvme/command/command.cc
)
set(SRC_HIL_NVME_COMMAND_ADMIN
  hil/nvme/command/admin/feature.cc
  hil/nvme/command/admin/log_page.cc
  hil/nvme/command/admin/delete_sq.cc
  hil/nvme/command/admin/create_sq.cc
  hil/nvme/command/admin/get_log_page.cc
  hil/nvme/command/admin/delete_cq.cc
  hil/nvme/command/admin/create_cq.cc
  hil/nvme/command/admin/identify.cc
  hil/nvme/command/admin/abort.cc
  hil/nvme/command/admin/set_feature.cc
  hil/nvme/command/admin/get_feature.cc
  hil/nvme/command/admin/async_event_request.cc
  hil/nvme/command/admin/namespace_management.cc
  hil/nvme/command/admin/namespace_attachment.cc
)
set(SRC_HIL_NVME_COMMAND_NVM
  hil/nvme/command/nvm/format_nvm.cc
  hil/nvme/command/nvm/flush.cc
  hil/nvme/command/nvm/write.cc
  hil/nvme/command/nvm/read.cc
  hil/nvme/command/nvm/compare.cc
  hil/nvme/command/nvm/dataset_management.cc
)
set(SRC_ICL
  icl/icl.cc
)
set(SRC_ICL_CACHE
  icl/cache/abstract_cache.cc
  icl/cache/generic_cache.cc
  icl/cache/ring_buffer.cc
  icl/cache/set_associative.cc
)
set(SRC_ICL_MANAGER
  icl/manager/generic_manager.cc
)
set(SRC_FTL
  ftl/def.cc
  ftl/filling.cc
  ftl/ftl.cc
  ftl/predictor_interface.cc
  ftl/ml-prediction/interface/predictor_impl.cc
  ftl/ml-prediction/interface/training_impl.cc
  ftl/ml-prediction/interface/predictor.cc
  ftl/ml-prediction/interface/workload_monitor.cc
  ftl/ml-prediction/interface/mlp_model.cc
)
set(SRC_FTL_ALLOC
  ftl/allocator/abstract_allocator.cc
  ftl/allocator/generic_allocator.cc
  ftl/allocator/victim_selection.cc
)
set(SRC_FTL_BACKGROUND_MANAGER
  ftl/background_manager/abstract_background_job.cc
  ftl/background_manager/basic_job_manager.cc
)
set(SRC_FTL_BASE
  ftl/base/abstract_ftl.cc
  ftl/base/page_level_ftl.cc
)
set(SRC_FTL_GC
  ftl/gc/abstract_gc.cc
  ftl/gc/advanced.cc
  ftl/gc/naive.cc
  ftl/gc/preemption.cc
)
set(SRC_FTL_MAPPING
  ftl/mapping/abstract_mapping.cc
  ftl/mapping/page_level_mapping.cc
)
set(SRC_FTL_READ_RECLAIM
  ftl/read_reclaim/abstract_read_reclaim.cc
  ftl/read_reclaim/basic_read_reclaim.cc
)
set(SRC_FTL_WEAR_LEVELING
  ftl/wear_leveling/abstract_wear_leveling.cc
  ftl/wear_leveling/static_wear_leveling.cc
)
set(SRC_FIL
  fil/fil.cc
)
set(SRC_FIL_NVM_PAL
  fil/nvm/pal/convert.cc
  fil/nvm/pal/Latency.cc
  fil/nvm/pal/LatencyMLC.cc
  fil/nvm/pal/LatencySLC.cc
  fil/nvm/pal/LatencyTLC.cc
  fil/nvm/pal/PAL2.cc
  fil/nvm/pal/PAL2_TimeSlot.cc
  fil/nvm/pal/PALStatistics.cc
  fil/nvm/pal/pal_wrapper.cc
)
set(SRC_FIL_SCHEDULER
  fil/scheduler/noop.cc
)
set(SRC_SIM
  sim/base_config.cc
  sim/config_reader.cc
  sim/config.cc
  sim/log.cc
  sim/simplessd.cc
)
set(SRC_UTIL
  util/bitset.cc
  util/disk.cc
  util/fifo.cc
  util/interface.cc
  util/sorted_map.cc
  util/stat_helper.cc
)

# Source group for MSVC
SOURCE_GROUP("Source Files\\cpu" FILES ${SRC_CPU})
SOURCE_GROUP("Source Files\\memory" FILES ${SRC_MEM})
SOURCE_GROUP("Source Files\\hil" FILES ${SRC_HIL})
SOURCE_GROUP("Source Files\\hil\\common" FILES ${SRC_HIL_COMMON})
SOURCE_GROUP("Source Files\\hil\\common" FILES ${SRC_HIL_NONE})
SOURCE_GROUP("Source Files\\hil\\nvme" FILES ${SRC_HIL_NVME})
SOURCE_GROUP("Source Files\\hil\\nvme\\command" FILES ${SRC_HIL_NVME_COMMAND})
SOURCE_GROUP("Source Files\\hil\\nvme\\command\\admin" FILES ${SRC_HIL_NVME_COMMAND_ADMIN})
SOURCE_GROUP("Source Files\\hil\\nvme\\command\\nvm" FILES ${SRC_HIL_NVME_COMMAND_NVM})
SOURCE_GROUP("Source Files\\icl" FILES ${SRC_ICL})
SOURCE_GROUP("Source Files\\icl\\cache" FILES ${SRC_ICL_CACHE})
SOURCE_GROUP("Source Files\\icl\\manager" FILES ${SRC_ICL_MANAGER})
SOURCE_GROUP("Source Files\\ftl" FILES ${SRC_FTL})
SOURCE_GROUP("Source Files\\ftl\\allocator" FILES ${SRC_FTL_ALLOC})
SOURCE_GROUP("Source Files\\ftl\\background_manager" FILES ${SRC_FTL_BACKGROUND_MANAGER})
SOURCE_GROUP("Source Files\\ftl\\base" FILES ${SRC_FTL_BASE})
SOURCE_GROUP("Source Files\\ftl\\gc" FILES ${SRC_FTL_GC})
SOURCE_GROUP("Source Files\\ftl\\mapping" FILES ${SRC_FTL_MAPPING})
SOURCE_GROUP("Source Files\\ftl\\read_reclaim" FILES ${SRC_FTL_READ_RECLAIM})
SOURCE_GROUP("Source Files\\ftl\\wear_leveling" FILES ${SRC_FTL_WEAR_LEVELING})
SOURCE_GROUP("Source Files\\fil" FILES ${SRC_FIL})
SOURCE_GROUP("Source Files\\fil\\nvm\\pal" FILES ${SRC_FIL_NVM_PAL})
SOURCE_GROUP("Source Files\\fil\\scheduler" FILES ${SRC_FIL_SCHEDULER})
SOURCE_GROUP("Source Files\\sim" FILES ${SRC_SIM} ${SRC_CONFIG})
SOURCE_GROUP("Source Files\\util" FILES ${SRC_UTIL})

# All sources with instruction statistics
set(SRCS
  ${SRC_HIL}
  ${SRC_HIL_COMMON}
  ${SRC_HIL_NONE}
  ${SRC_HIL_NVME}
  ${SRC_HIL_NVME_COMMAND}
  ${SRC_HIL_NVME_COMMAND_ADMIN}
  ${SRC_HIL_NVME_COMMAND_NVM}
  ${SRC_ICL}
  ${SRC_ICL_CACHE}
  ${SRC_ICL_MANAGER}
  ${SRC_FTL}
  ${SRC_FTL_ALLOC}
  ${SRC_FTL_BACKGROUND_MANAGER}
  ${SRC_FTL_BASE}
  ${SRC_FTL_GC}
  ${SRC_FTL_MAPPING}
  ${SRC_FTL_READ_RECLAIM}
  ${SRC_FTL_WEAR_LEVELING}
  ${SRC_FIL}
  ${SRC_FIL_NVM_PAL}
  ${SRC_FIL_SCHEDULER}
)

# All sources without instruction statistics
set(SRCS_EXCLUDE
  ${SRC_CPU}
  ${SRC_MEM}
  ${SRC_SIM}
  ${SRC_UTIL}
  ${SRC_CONFIG}
  ${OUTPUT_VERSION_FILE}
)

if(MSVC AND FW_LATENCY)
  message(WARNING "Automatic firmware latency generation is not tested on Windows")
endif()

if(FW_LATENCY)
  if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message(STATUS "Force release build")
    set(uppercase_CMAKE_BUILD_TYPE "RELEASE")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Force release build" FORCE)
  endif()

  # Check llvm-simplessd is valid
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/lib/llvm-simplessd/CMakeLists.txt")
    message(FATAL_ERROR "Failed to find llvm-simplessd library. \
    Check you initialized submodules.")
  endif()

  # Detects CROSS_COMPILE
  if(NOT "${FW_CROSS_COMPILE}" STREQUAL "")
    message("-- Detecting include path from ${FW_CROSS_COMPILE}g++")

    # Get include paths
    execute_process(COMMAND sh -c "${FW_CROSS_COMPILE}g++ -E -Wp,-v -x c++ - < /dev/null"
      RESULT_VARIABLE FW_CROSS_RESULT
      ERROR_VARIABLE FW_CROSS_OUTPUT
      OUTPUT_QUIET
    )

    if(NOT ${FW_CROSS_RESULT} EQUAL 0)
      message(FATAL_ERROR "Command failed with error code: ${FW_CROSS_RESULT}")
    endif()

    # Parse
    string(REPLACE "\n" ";" FW_CROSS_OUTPUT ${FW_CROSS_OUTPUT})

    set(GOOD 0)
    set(FW_INCLUDE "")

    foreach(LINE IN ITEMS ${FW_CROSS_OUTPUT})
      if("${LINE}" STREQUAL "End of search list.")
        set(GOOD 0)
      endif()

      if(${GOOD} EQUAL 1)
        string(STRIP "${LINE}" LINE)
        list(APPEND FW_INCLUDE "-I${LINE}")
      endif()

      if("${LINE}" STREQUAL "#include <...> search starts here:")
        set(GOOD 1)
      endif()
    endforeach()

    if("${FW_TARGET}" STREQUAL "")
      message(FATAL_ERROR "You need to specify FW_TARGET if you use FW_CROSS_COMPILE")
    endif()

    if("${FW_CPU}" STREQUAL "")
      message(FATAL_ERROR "You need to specify FW_CPU if you use FW_CROSS_COMPILE")
    endif()
  endif()

  # Make OBJS list
  set(BUILD_DIRECTORY "${PROJECT_BINARY_DIR}/objects")
  string(REGEX REPLACE "([^;]+)" "${BUILD_DIRECTORY}/\\1.o" OBJS "${SRCS}")

  # Define target
  add_library(simplessd-objects OBJECT IMPORTED)
  set_target_properties(simplessd-objects PROPERTIES IMPORTED_OBJECTS "${OBJS}")

  # We need to build LLVM pass
  add_subdirectory(${PROJECT_SOURCE_DIR}/lib/llvm-simplessd)

  # Make commands
  set(LLVM_PASS "${CMAKE_BINARY_DIR}/libllvm-simplessd.so")
  set(STAT_COLLECTOR "${CMAKE_BINARY_DIR}/inststat-generator")

  list(LENGTH SRCS SRC_LEN)
  math(EXPR SRC_LEN "${SRC_LEN} - 1")

  # Generate clang++ arguments
  set(FW_DEFAULT_ARGS
    "-std=c++17"
    "-g"
    "-c"
    "-emit-llvm"
    "-I${PROJECT_SOURCE_DIR}"
    "-I${DRAMPOWER_SOURCE_DIR}"
    "-DUSE_WRITE_THROUGH=1"
  )

  set(FW_FIRST_PASS_ARGS
    "-DEXCLUDE_CPU"
  )

  if(NOT "${FW_TARGET}" STREQUAL "")
    string(FIND ${FW_TARGET} "-" FW_TARGET_POS)
    string(SUBSTRING ${FW_TARGET} 0 ${FW_TARGET_POS} FW_ARCH)

    list(APPEND FW_FIRST_PASS_ARGS "--target=${FW_TARGET}")
    list(APPEND FW_FIRST_PASS_ARGS "-m${FW_ARCH}")
  endif()

  if(NOT "${FW_CPU}" STREQUAL "")
    set(FW_CPU_ARG "-mcpu=${FW_CPU}")
    list(APPEND FW_FIRST_PASS_ARGS "${FW_CPU_ARG}")
  endif()

  foreach(IDX RANGE ${SRC_LEN})
    list(GET SRCS ${IDX} SRC)
    list(GET OBJS ${IDX} OBJ)

    get_filename_component(DIR ${OBJ} DIRECTORY)

    add_custom_command(OUTPUT ${OBJ}
      COMMAND mkdir -p ${DIR}
      COMMAND clang++ ${FW_DEFAULT_ARGS} ${FW_FIRST_PASS_ARGS} ${FW_INCLUDE} -o "${OBJ}.bc" ${SRC}
      COMMAND opt -enable-new-pm=0 --load=${LLVM_PASS} --blockcollector -O2 -o "${OBJ}.opt.bc" "${OBJ}.bc"
      COMMAND llc ${FW_CPU_ARG} -O2 -filetype=asm -o "${OBJ}.bc.S" "${OBJ}.opt.bc"
      COMMAND ${STAT_COLLECTOR} "${OBJ}.bc"
      COMMAND clang++ ${FW_DEFAULT_ARGS} -O2 -o "${OBJ}.bc" ${SRC}
      COMMAND opt -enable-new-pm=0 --load=${LLVM_PASS} --inststat -O2 -o "${OBJ}.opt.bc" "${OBJ}.bc"
      COMMAND llc -O2 -filetype=obj -o ${OBJ} "${OBJ}.opt.bc"
      DEPENDS ${SRC}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMENT "Instrumenting file ${SRC}"
    )
  endforeach()

  # CMake should run custom commands to create simplessd-objects
  add_custom_target(build-objects ALL
    DEPENDS ${OBJS}
  )

  # Make target
  add_library(simplessd STATIC
    ${SRCS_EXCLUDE}
    $<TARGET_OBJECTS:simplessd-objects>
  )

  # Make dependency
  add_dependencies(build-objects llvm-simplessd)
  add_dependencies(simplessd build-objects)
else()
  add_library(simplessd STATIC ${SRCS} ${SRCS_EXCLUDE})
endif()

# Configure test
set(SRC_TEST
  test/main.cc
  test/config_reader.cc
  test/sorted_map.cc
)

add_executable(test_simplessd ${SRC_TEST})

# Always use write-through cache
target_compile_definitions(simplessd PRIVATE -DUSE_WRITE_THROUGH=1)

# Platform specific settings
if(MSVC)
  target_compile_definitions(simplessd PRIVATE -D_CRT_SECURE_NO_WARNINGS)
  target_compile_options(simplessd PRIVATE /wd4819) # Surpress unicode warning
else()
  if(USE_SANITIZER)
    set(SANFLAG
      -fsanitize=address
      -fsanitize=undefined
      -fno-sanitize-recover=all
      -fsanitize=float-divide-by-zero
      -fsanitize=float-cast-overflow
      -fno-sanitize=null
      -fno-sanitize=alignment
    )
  else()
    set(SANFLAG "")
  endif()

  target_compile_options(simplessd PRIVATE -g -Wall -Wextra -Werror -Wno-attributes ${SANFLAG})

  set(CMAKE_CXX_FLAGS "-D__FILENAME__='\"$(subst ${PROJECT_SOURCE_DIR}/,,$(abspath $<))\"' ${CMAKE_CXX_FLAGS}")

  target_link_libraries(simplessd stdc++fs ${SANFLAG})
  target_link_libraries(test_simplessd stdc++fs)

  if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message(STATUS "Using debug build")

    target_compile_definitions(simplessd PRIVATE -DSIMPLESSD_DEBUG)
  endif()

  if(FW_LATENCY)
    target_compile_definitions(simplessd PRIVATE -DFW_LATENCY=1)
  endif()
endif()

# Force version checking in every build
add_dependencies(simplessd ${VERSION_TARGET})

set(CMAKE_PREFIX_PATH "/root/parallelism_workspace/SimpleSSD-Standalone/libtorch")
find_package(Torch REQUIRED)

target_link_libraries(simplessd "${TORCH_LIBRARIES}")

target_link_libraries(simplessd mcpat pugixml nlohmann_json::nlohmann_json)
target_link_libraries(test_simplessd simplessd Catch2::Catch2)

include(CTest)
include(${PROJECT_SOURCE_DIR}/lib/catch2/contrib/Catch.cmake)

catch_discover_tests(test_simplessd)
