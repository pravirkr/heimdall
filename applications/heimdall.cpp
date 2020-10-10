/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>

#include <fmt/format.h>

#include <utils/stopwatch.hpp>
#include <utils/parse_command_line.hpp>
#include <hd/default_params.hpp>
#include <hd/pipeline.hpp>
#include <hd/error.hpp>

// input formats supported
#include <data_types/data_source.hpp>
#include <data_types/filterbank.hpp>

int main(int argc, char* argv[]) {
    hd_params params;
    hd_set_default_params(&params);
    int ok = hd_parse_command_line(argc, argv, &params);

    if (ok < 0)
        return 1;

    DataSource* data_source = 0;

    // Read from filterbank file
    try {
        data_source = new SigprocFile(params.sigproc_file, params.fswap);
    } catch (std::exception& ex) {
        fmt::print(stderr, "FILE ERROR: Failed to open data file: {}\n",
            ex.what());
        return -1;
    }

    if (!params.override_beam) {
        if (data_source->get_beam() > 0)
            params.beam = data_source->get_beam() - 1;
        else
            params.beam = 0;
    }

    params.f0                 = data_source->get_f0();
    params.df                 = data_source->get_df();
    params.dt                 = data_source->get_tsamp();
    params.nchans             = data_source->get_nchan();
    params.utc_start          = data_source->get_utc_start();
    params.spectra_per_second = data_source->get_spectra_rate();

    size_t nsamps_gulp = params.nsamps_gulp;
    size_t stride      = data_source->get_stride();
    size_t nbits       = data_source->get_nbit();

    // ideally this should be nsamps_gulp + max overlap, but just do x2
    size_t filterbank_bytes = 2 * nsamps_gulp * stride;
    if (params.verbosity >= 2){
        fmt::print("Allocating filterbank data vector for {} samples with size {} bytes\n",
            nsamps_gulp, filterbank_bytes);
    }
    std::vector<hd_byte> filterbank(filterbank_bytes);

    // Create the pipeline object
    // --------------------------
    hd_pipeline pipeline;
    hd_error    error;
    error = hd_create_pipeline(&pipeline, params);
    if (error != HD_NO_ERROR) {
        fmt::print(stderr, "ERROR: Pipeline creation failed:");
        fmt::print(stderr, "\t\t{}\n", hd_get_error_string(error));
        return -1;
    }
    // --------------------------

    if (params.verbosity >= 1) {
        fmt::print("Beginning data processing, requesting {} samples\n", 
            nsamps_gulp);
    }

    // start a timer for the whole pipeline
    // Stopwatch pipeline_timer;

    bool   stop_requested = false;
    size_t total_nsamps   = 0;
    size_t nsamps_read =
        data_source->get_data(nsamps_gulp, (char*)&filterbank[0]);
    size_t overlap = 0;
    while (nsamps_read && !stop_requested) {
        if (params.verbosity >= 1) {
            fmt::print("Executing pipeline on new gulp of {} samples...\n",
                nsamps_read);
        }
        // pipeline_timer.start();

        if (params.verbosity >= 2) {
            fmt::print(" nsamp_gulp={} overlap={} nsamps_read={} nsamps_read+overlap={}\n",
                nsamps_gulp, overlap, nsamps_read, nsamps_read + overlap);
        }

        hd_size nsamps_processed;
        error = hd_execute(pipeline,
                           &filterbank[0],
                           nsamps_read + overlap,
                           nbits,
                           total_nsamps,
                           &nsamps_processed);
        if (error == HD_NO_ERROR) {
            if (params.verbosity >= 1){
                fmt::print("Processed {} samples\n", nsamps_processed);
            }
        } else if (error == HD_TOO_MANY_EVENTS) {
            if (params.verbosity >= 1){
                fmt::print(stderr, 
                    "WARNING: hd_execute produces too many events, some "
                    "data skipped\n");
            }
        } else {
            fmt::print(stderr, "ERROR: Pipeline execution failed:");
            fmt::print(stderr, "\t\t{}\n", hd_get_error_string(error));
            hd_destroy_pipeline(pipeline);
            return -1;
        }

        if (params.verbosity >= 1){
            fmt::print("Main: nsamps_processed={}\n", nsamps_processed);
        }

        // pipeline_timer.stop();
        // float tsamp = data_source->get_tsamp() / 1000000;
        // cout << "pipeline time: " << pipeline_timer.getTime() << " of " <<
        // (nsamps_read+overlap) * tsamp << endl; pipeline_timer.reset();

        total_nsamps += nsamps_processed;
        // Now we must 'rewind' to do samples that couldn't be processed
        // Note: This assumes nsamps_gulp > 2*overlap
        std::copy(&filterbank[nsamps_processed * stride],
                  &filterbank[(nsamps_read + overlap) * stride],
                  &filterbank[0]);
        overlap += nsamps_read - nsamps_processed;
        nsamps_read = data_source->get_data(
            nsamps_gulp, (char*)&filterbank[overlap * stride]);

        // at the end of data, never execute the pipeline
        if (nsamps_read < nsamps_gulp)
            stop_requested = 1;
    }

    // final iteration for nsamps which is not a multiple of gulp size - overlap
    if (stop_requested && nsamps_read > 0) {
        if (params.verbosity >= 1){
            fmt::print("Final sub gulp: nsamps_read={} nsamps_gulp={} overlap={}\n",
                nsamps_read, nsamps_gulp, overlap);
        }
        hd_size nsamps_processed;
        hd_size nsamps_to_process = nsamps_read + overlap;
        if (nsamps_to_process > nsamps_gulp)
            nsamps_to_process = nsamps_gulp;
        error = hd_execute(pipeline,
                           &filterbank[0],
                           nsamps_to_process,
                           nbits,
                           total_nsamps,
                           &nsamps_processed);
        if (params.verbosity >= 1){
            fmt::print("Final sub gulp: nsamps_processed={}\n", nsamps_processed);
        }

        if (error == HD_NO_ERROR) {
            if (params.verbosity >= 1){
                fmt::print("Processed {} samples.\n", nsamps_processed);
            }
        } else if (error == HD_TOO_MANY_EVENTS) {
            if (params.verbosity >= 1){
                fmt::print(stderr, 
                    "WARNING: hd_execute produces too many events, some data skipped\n");
            }
        } else if (error == HD_TOO_FEW_NSAMPS) {
            if (params.verbosity >= 1){
                fmt::print(stderr, 
                    "WARNING: hd_execute did not have enough samples to process\n");
            }
        } else {
            fmt::print(stderr, "ERROR: Pipeline execution failed:");
            fmt::print(stderr, "\t\t{}\n", hd_get_error_string(error));
        }
        total_nsamps += nsamps_processed;
    }

    if (params.verbosity >= 1) {
        fmt::print("Successfully processed a total of {} samples.\n", 
            total_nsamps);
    }

    if (params.verbosity >= 1) {
        fmt::print("Shutting down...\n");
    }

    hd_destroy_pipeline(pipeline);

    if (params.verbosity >= 1) {
        fmt::print("All done.\n");
    }
}
