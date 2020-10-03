/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <data_types/header.hpp>
#include <utils/exceptions.hpp>

// Internal functions
namespace detail {
// Note: This is an internal function and is not intended to be used externally
void write_time_series_header(std::ofstream& out_file,
                              size_t         nbits,
                              float          dt,
                              float          dm) {
    // Write the required header information
    header_write(out_file, "HEADER_START");
    header_write(out_file, "data_type", 2);
    header_write(out_file, "nchans", 1);
    header_write(out_file, "nifs", 1);
    header_write(out_file, "refdm", dm);
    header_write(out_file, "nbits", (int)nbits);
    header_write(out_file, "tsamp", dt);
    header_write(out_file, "HEADER_END");
}

void write_filterbank_header(std::ofstream& out_file,
                             size_t         nbits,
                             int            nchan,
                             float          dt,
                             float          f0,
                             float          df) {
    // Write the required header information
    header_write(out_file, "HEADER_START");
    header_write(out_file, "data_type", 1);
    header_write(out_file, "nchans", nchan);
    header_write(out_file, "nifs", 1);
    header_write(out_file, "tsamp", dt);
    header_write(out_file, "nbits", (int)nbits);
    header_write(out_file, "fch1", f0);
    header_write(out_file, "foff", df);
    header_write(out_file, "HEADER_END");
}
}  // namespace detail

// Float data type
void write_host_time_series(const float* data,
                            size_t       nsamps,
                            float        dt,
                            float        dm,
                            std::string  filename) {
    // Open the output file and write the data
    std::ofstream file(filename.c_str(), std::ios::binary);
    detail::write_time_series_header(file, 32, dt, dm);
    size_t size_bytes = nsamps * sizeof(float);
    file.write((char*)data, size_bytes);
    file.close();
}

void write_device_time_series(const float* data,
                              size_t       nsamps,
                              float        dt,
                              float        dm,
                              std::string  filename) {
    std::vector<float> h_data(nsamps);
    cudaMemcpy(&h_data[0], data, nsamps * sizeof(float),
               cudaMemcpyDeviceToHost);
    ErrorChecker::check_cuda_error("Error from cudaMemcpy");
    write_host_time_series(&h_data[0], nsamps, dt, dm, filename);
}

// Integer data type
void write_host_time_series(const unsigned int* data,
                            size_t              nsamps,
                            size_t              nbits,
                            float               dt,
                            float               dm,
                            std::string         filename) {
    // Here we convert the data to floats before writing the data
    std::vector<float> float_data(nsamps);
    switch (nbits) {
        case sizeof(char) * 8:
            for (int i = 0; i < (int)nsamps; ++i)
                float_data[i] = (float)((unsigned char*)data)[i];
            break;
        case sizeof(short) * 8:
            for (int i = 0; i < (int)nsamps; ++i)
                float_data[i] = (float)((unsigned short*)data)[i];
            break;
        case sizeof(int) * 8:
            for (int i = 0; i < (int)nsamps; ++i)
                float_data[i] = (float)((unsigned int*)data)[i];
            break;
        default:
            // Unpack to float
            size_t mask = (1 << nbits) - 1;
            size_t spw  = sizeof(unsigned int) * 8 / nbits;  // Samples per word
            for (int i = 0; i < (int)nsamps; ++i)
                float_data[i] = (data[i / spw] >> (i % spw * nbits)) & mask;
    }
    write_host_time_series(&float_data[0], nsamps, dt, dm, filename);
}

void write_device_time_series(const unsigned int* data,
                              size_t              nsamps,
                              size_t              nbits,
                              float               dt,
                              float               dm,
                              std::string         filename) {
    size_t nsamps_words = nsamps * nbits / (sizeof(unsigned int) * 8);
    std::vector<unsigned int> h_data(nsamps_words);
    cudaMemcpy(&h_data[0], data, nsamps_words * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    ErrorChecker::check_cuda_error("Error from cudaMemcpy");
    write_host_time_series(&h_data[0], nsamps, nbits, dt, dm, filename);
}

void write_host_filterbank(const hd_byte* filterbank,
                           int            nchan,
                           int            nsamps,
                           int            nbits,
                           float          dt,
                           float          f0,
                           float          df,
                           std::string    filename) {
    // Open the output file and write the data
    std::ofstream file(filename.c_str(), std::ios::binary);
    detail::write_filterbank_header(file, nbits, nchan, dt, f0, df);
    size_t size_bytes = nchan * nsamps * nbits / 8;
    file.write((char*)&filterbank[0], size_bytes);
    file.close();
}
