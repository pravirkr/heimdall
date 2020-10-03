#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

class ErrorChecker {
public:
    static void check_file_error(std::ifstream& m_file_stream,
                                 std::string    filename) {
        if (!m_file_stream.good()) {
            std::stringstream error_msg;
            error_msg << "File " << filename << " could not be opened: ";

            if ((m_file_stream.rdstate() & std::ifstream::failbit) != 0)
                error_msg << "Logical error on i/o operation" << std::endl;

            if ((m_file_stream.rdstate() & std::ifstream::badbit) != 0)
                error_msg << "Read/writing error on i/o operation" << std::endl;

            if ((m_file_stream.rdstate() & std::ifstream::eofbit) != 0)
                error_msg << "End-of-File reached on input operation"
                          << std::endl;

            throw std::runtime_error(error_msg.str());
        }
    }

    static void check_file_error(std::ofstream& m_file_stream,
                                 std::string    filename) {
        if (!m_file_stream.good()) {
            std::stringstream error_msg;
            error_msg << "File " << filename << " could not be opened: ";

            if ((m_file_stream.rdstate() & std::ifstream::failbit) != 0)
                error_msg << "Logical error on i/o operation" << std::endl;

            if ((m_file_stream.rdstate() & std::ifstream::badbit) != 0)
                error_msg << "Read/writing error on i/o operation" << std::endl;

            if ((m_file_stream.rdstate() & std::ifstream::eofbit) != 0)
                error_msg << "End-of-File reached on input operation"
                          << std::endl;

            throw std::runtime_error(error_msg.str());
        }
    }

    static void check_cuda_error(std::string msg = "Unspecified location") {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::stringstream error_msg;
            error_msg << "CUDA failed with error: " << cudaGetErrorString(error)
                      << std::endl
                      << "Additional: " << msg << std::endl;
            throw std::runtime_error(error_msg.str());
        }
    }
};