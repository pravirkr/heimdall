cmake_minimum_required(VERSION 3.15)

project(heimdall VERSION 1.0.1 LANGUAGES C CXX)

set(BOOST_ROOT "/apps/skylake/software/mpi/gcc/6.4.0/openmpi/3.0.0/boost/1.66.0")
set(CUDACXX $ENV{CUDA_PATH})
set(DEDISP "dedisp")

find_package(Boost 1.60 REQUIRED COMPONENTS filesystem)
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
    message(STATUS "No CUDA compiler found...")
endif(CMAKE_CUDA_COMPILER)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_FLAGS "-c -Xcompiler=-Wall -O3 -arch=sm_60 -Wno-deprecated-declarations")


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
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} ${DEDISP} pthread)

install(TARGETS ${PROJECT_NAME} 
    DESTINATION ${PROJECT_SOURCE_DIR}/bin
)


add_subdirectory("tests")




