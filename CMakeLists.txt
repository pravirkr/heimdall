cmake_minimum_required(VERSION 3.15)

project(heimdall VERSION 1.0.1 LANGUAGES C CXX)

set(ENV{CUDACXX} $ENV{CUDA_PATH}/bin/nvcc)
set(DEDISP "dedisp")
set(FMTLIB "fmtlib")

option(ENABLE_CLANG_TIDY "Run Clang Tidy to get static analysis" OFF)

include(FindGit)
find_package(Git)

if (NOT Git_FOUND)
    message(FATAL_ERROR "Git not found!")
endif ()

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
set(CMAKE_CXX_FLAGS "-Wall")

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_FLAGS "-c -Xcompiler=-Wall -O3 -arch=sm_60 -Wno-deprecated-declarations")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include(FetchContent)

FetchContent_Declare(
    fmtlib
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        7.0.3
)

FetchContent_GetProperties(fmtlib)
if(NOT fmtlib_POPULATED)
    FetchContent_Populate(fmtlib)
    message(STATUS "fmt source dir: ${fmtlib_SOURCE_DIR}")
    message(STATUS "fmt binary dir: ${fmtlib_BINARY_DIR}")
    add_subdirectory(${fmtlib_SOURCE_DIR} ${fmtlib_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

message(STATUS "fmt include dir: ${fmtlib_SOURCE_DIR}/include")

# ------------------------------------------------------------------------------
# Clang Tidy
# ------------------------------------------------------------------------------
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
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${DEDISP} pthread)
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt-header-only)

install(TARGETS ${PROJECT_NAME} 
    DESTINATION ${PROJECT_SOURCE_DIR}/bin
)


add_subdirectory("tests")




