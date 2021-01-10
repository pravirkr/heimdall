cmake_minimum_required(VERSION 3.15)

project(heimdall VERSION 1.0.1 LANGUAGES CXX)

option(ENABLE_CLANG_TIDY "Run Clang Tidy to get static analysis" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(DEDISP "dedisp")

include(FindGit)
include(FetchContent)

find_package(Git)
if (NOT Git_FOUND)
    message(FATAL_ERROR "Git not found!")
endif ()

set(ENV{CUDACXX} $ENV{CUDA_PATH}/bin/nvcc)
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL 10.0.0)
        message(FATAL_ERROR "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}. Need CUDA 10+")
    else()
        message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")
    endif()
else(CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA compiler found...")
endif(CMAKE_CUDA_COMPILER)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_FLAGS "-c -Xcompiler=-Wall -arch=sm_60 -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# search for CMake configuration file for fmt
find_package(fmt 7.1.2 QUIET)
if(fmt_FOUND)
    message(STATUS "Found {fmt}: ${fmt_DIR} (found version '${fmt_VERSION}')")
else(fmt_FOUND)
    message(STATUS "No local {fmt} installation found. Downloading from Git...")
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        7.1.2
    )
    FetchContent_GetProperties(fmt)
    if(NOT fmt_POPULATED)
        FetchContent_Populate(fmt)
        message(STATUS "{fmt} source dir: ${fmt_SOURCE_DIR}")
        add_subdirectory(${fmt_SOURCE_DIR} ${fmt_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    message(STATUS "{fmt} include dir: ${fmt_SOURCE_DIR}/include")
endif(fmt_FOUND)


# dedisp shared library
add_library(${DEDISP} SHARED ${PROJECT_SOURCE_DIR}/src/dedisp.cu)
set_target_properties(${DEDISP}
    PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_include_directories(${DEDISP} PUBLIC ${PROJECT_SOURCE_DIR}/include)

install(TARGETS ${DEDISP}
    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
)


# heimdall
add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/applications/heimdall.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline.cu
    ${PROJECT_SOURCE_DIR}/src/clean_filterbank_rfi.cu
    ${PROJECT_SOURCE_DIR}/src/error.cpp
    ${PROJECT_SOURCE_DIR}/src/find_giants.cu
    ${PROJECT_SOURCE_DIR}/src/get_rms.cu
    ${PROJECT_SOURCE_DIR}/src/label_candidate_clusters.cu
    ${PROJECT_SOURCE_DIR}/src/matched_filter.cu
    ${PROJECT_SOURCE_DIR}/src/measure_bandpass.cu
    ${PROJECT_SOURCE_DIR}/src/median_filter.cu
    ${PROJECT_SOURCE_DIR}/src/merge_candidates.cu
    ${PROJECT_SOURCE_DIR}/src/remove_baseline.cu
)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${fmt_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} PRIVATE ${DEDISP} pthread fmt::fmt-header-only)

install(TARGETS ${PROJECT_NAME}
    DESTINATION ${PROJECT_SOURCE_DIR}/bin
)


add_subdirectory("tests")


# ------------------------------------------------------------------------------
# Clang Tidy
# ------------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_BIN clang-tidy)
    find_program(RUN_CLANG_TIDY_BIN run-clang-tidy)

    if(CLANG_TIDY_BIN STREQUAL "CLANG_TIDY_BIN-NOTFOUND")
        message(FATAL_ERROR "unable to locate clang-tidy")
    endif()

    if(RUN_CLANG_TIDY_BIN STREQUAL "RUN_CLANG_TIDY_BIN-NOTFOUND")
        message(FATAL_ERROR "unable to locate run-clang-tidy")
    endif()

    list(APPEND RUN_CLANG_TIDY_BIN_ARGS
        -clang-tidy-binary ${CLANG_TIDY_BIN}
        -header-filter=.*
        -checks=clan*,cert*,misc*,perf*,cppc*,read*,mode*,-cert-err58-cpp,-misc-noexcept-move-constructor
    )

    add_custom_target(
        tidy
        COMMAND ${RUN_CLANG_TIDY_BIN} ${RUN_CLANG_TIDY_BIN_ARGS}
        COMMENT "running clang tidy"
    )

endif()
