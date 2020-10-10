/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <iostream>
#include <fstream>
#include <cfloat>
#include <stdexcept>
#include <data_types/header.hpp>
#include <data_types/data_source.hpp>
#include <utils/exceptions.hpp>

class SigprocFile : public DataSource {
private:
    inline unsigned char flip_4bit(unsigned char val) {
        return (((val & 0x3) << 6) | ((val & 0xC) << 2) | ((val & 0x30) >> 2) |
                ((val & 0xC0) >> 6));
    };

    inline unsigned char flip_2bit(unsigned char val) {
        return (((val & 0x0F) << 4) | ((val & 0xF0) >> 4));
    };

    std::ifstream m_file_stream;
    int           m_error;
    bool          first_time;
    float         offset;
    float         scale;
    bool          fswap;

public:
    SigprocFile(std::string filename, bool _fswap)
        : m_file_stream(filename, std::ios::binary), DataSource() {

        fswap = _fswap;
        ErrorChecker::check_file_error(m_file_stream, filename);
        SigprocHeader m_header;
        read_header(m_file_stream, m_header);
        ErrorChecker::check_file_error(m_file_stream, filename);

        nchan        = m_header.nchans;
        nbit         = m_header.nbits;
        beam         = m_header.ibeam;
        tsamp        = m_header.tsamp;  // in seconds
        spectra_rate = 1 / tsamp;

        f0 = m_header.fch1;
        df = m_header.foff;

        utc_start = mjd2utctm(m_header.tstart);
        char buffer[128];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H:%M:%S", std::localtime(&utc_start));

        stride     = (nchan * nbit) / (8 * sizeof(char));
        first_time = true;
        offset     = 0;
        scale      = 1;
    }

    ~SigprocFile() { m_file_stream.close(); }

    size_t get_data(size_t nsamps, char* data) {
        size_t nchan_bytes = stride;
        m_file_stream.read((char*)&data[0], nsamps * nchan_bytes);

        // by default sigproc 32-bit data are stored as floats, dedisp requires
        // unsigned ints
        if (nbit == 32) {
            const unsigned nfloats = nsamps * nchan;
            unsigned*      out     = (unsigned*)&data[0];
            float*         in      = (float*)&data[0];

            if (first_time) {
                float sum       = 0;
                float min_float = FLT_MAX;
                float max_float = -FLT_MAX;
                for (unsigned i = 0; i < nfloats; i++) {
                    if (in[i] < min_float)
                        min_float = in[i];
                    if (in[i] > max_float)
                        max_float = in[i];
                    sum += in[i];
                }
                float mean = sum / nfloats;
                offset     = (2 ^ 31) - mean;
                scale      = (max_float - min_float) / (2 ^ 28);
                first_time = false;
            }

            for (unsigned i = 0; i < nfloats; i++) {
                out[i] = (unsigned int)((in[i] / scale) + offset);
            }
        }

        if (fswap) {
            if (nbit == 2 || nbit == 4 || nbit == 8) {
                unsigned samps_per_char   = 8 / nbit;
                char*    inptr            = data;
                unsigned nchar_per_sample = nchan / samps_per_char;
                for (unsigned i = 0; i < nsamps; i++) {
                    for (unsigned idat = 0; idat < (nchar_per_sample / 2);
                         idat++) {
                        unsigned      odat     = nchar_per_sample - (1 + idat);
                        unsigned char this_dat = inptr[idat];
                        unsigned char that_dat = inptr[odat];

                        if (nbit == 2) {
                            this_dat = flip_2bit(this_dat);
                            that_dat = flip_2bit(that_dat);
                        }
                        if (nbit == 4) {
                            this_dat = flip_4bit(this_dat);
                            that_dat = flip_4bit(that_dat);
                        }

                        inptr[odat] = this_dat;
                        inptr[idat] = that_dat;
                    }
                    inptr += nchar_per_sample;
                }
            } else
                throw std::runtime_error(
                    "Could not FSWAP on the input bitrate");
        }

        size_t bytes_read = m_file_stream.gcount();
        return bytes_read / nchan_bytes;
    };
};
