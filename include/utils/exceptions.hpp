#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

class ErrorChecker {
public:
    template <class Tstream>
    static void check_file_error(Tstream& stream, std::string filename) {
        if (!stream.good()) {
            std::string file_msg;

            if ((stream.rdstate() & Tstream::failbit) != 0)
                file_msg = fmt::format("Logical error on i/o operation");

            if ((stream.rdstate() & Tstream::badbit) != 0)
                file_msg = fmt::format("Read/writing error on i/o operation");

            if ((stream.rdstate() & Tstream::eofbit) != 0)
                file_msg
                    = fmt::format("End-of-File reached on input operation");

            std::string error_msg = fmt::format(
                "File {} could not be opened: {}\n", filename, file_msg);

            throw std::runtime_error(error_msg);
        }
    }

    static void check_cuda_error(cudaError_t error,
                                 std::string msg = "Unspecified location") {
        if (error != cudaSuccess) {
            std::string error_msg
                = fmt::format("CUDA failed with error: {}\n\tAdditional: {}\n",
                              cudaGetErrorString(error), msg);
            throw std::runtime_error(error_msg);
        }
    }

    static void check_cuda_error(std::string msg = "Unspecified location") {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::string error_msg
                = fmt::format("CUDA failed with error: {}\n\tAdditional: {}\n",
                              cudaGetErrorString(error), msg);
            throw std::runtime_error(error_msg);
        }
    }

    static void throw_error(std::string msg) { throw std::runtime_error(msg); }
};