/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <hd/params.hpp>
#include <hd/default_params.hpp>

void hd_print_usage() {
    hd_params p;
    hd_set_default_params(&p);

    std::cout << "Usage: heimdall [options]" << std::endl;
    std::cout << "    -f  filename             process specified SIGPROC "
                 "filterbank file"
              << std::endl;
    std::cout << "    -vVgG                    increase verbosity level"
              << std::endl;
    std::cout << "    -yield_cpu               yield CPU during GPU operations"
              << std::endl;
    std::cout << "    -gpu_id ID               run on specified GPU"
              << std::endl;
    std::cout << "    -nsamps_gulp num         number of samples to be read at "
                 "a time ["
              << p.nsamps_gulp << "]" << std::endl;
    std::cout << "    -baseline_length num     number of seconds over which to "
                 "smooth the baseline ["
              << p.baseline_length << "]" << std::endl;
    std::cout << "    -beam ##                 over-ride beam number"
              << std::endl;
    std::cout << "    -output_dir path         create all output files in "
                 "specified path"
              << std::endl;
    std::cout << "    -dm min max              min and max DM" << std::endl;
    std::cout << "    -dm_tol num              SNR loss tolerance between each "
                 "DM trial ["
              << p.dm_tol << "]" << std::endl;
    std::cout << "    -zap_chans start end     zap all channels between start "
                 "and end channels inclusive"
              << std::endl;
    std::cout << "    -max_giant_rate nevents  limit the maximum number of "
                 "individual detections per minute to nevents"
              << std::endl;
    std::cout << "    -dm_pulse_width num      expected intrinsic width of the "
                 "pulse signal in microseconds"
              << std::endl;
    std::cout << "    -dm_nbits num            number of bits per sample in "
                 "dedispersed time series ["
              << p.dm_nbits << "]" << std::endl;
    std::cout << "    -no_scrunching           don't use an adaptive time "
                 "scrunching during dedispersion"
              << std::endl;
    std::cout << "    -scrunching_tol num      smear tolerance factor for time "
                 "scrunching ["
              << p.scrunch_tol << "]" << std::endl;
    std::cout << "    -rfi_tol num             RFI exicision threshold limits ["
              << p.rfi_tol << "]" << std::endl;
    std::cout << "    -rfi_no_narrow           disable narrow band RFI excision"
              << std::endl;
    std::cout << "    -rfi_no_broad            disable 0-DM RFI excision"
              << std::endl;
    std::cout << "    -rfi_boxcar_max num      maximum boxcar width in samples "
                 "while 0-DM RFI excision ["
              << p.rfi_boxcar_max << "]" << std::endl;
    std::cout
        << "    -boxcar_max num          maximum boxcar width in samples ["
        << p.boxcar_max << "]" << std::endl;
    std::cout << "    -detect_thresh num       Detection threshold (units of "
                 "std. dev.) ["
              << p.detect_thresh << "]" << std::endl;
    std::cout << "    -cand_sep_time num       Min separation between "
                 "candidates (in samples) ["
              << p.cand_sep_time << "]" << std::endl;
    std::cout << "    -cand_sep_filter num     Min separation between "
                 "candidates (in filters) ["
              << p.cand_sep_filter << "]" << std::endl;
    std::cout << "    -cand_sep_dm_trial num   Min separation between "
                 "candidates (in DM trials) ["
              << p.cand_sep_dm << "]" << std::endl;
    std::cout << "    -cand_rfi_dm_cut num     Minimum DM for valid candidate ["
              << p.cand_rfi_dm_cut << "]" << std::endl;
    std::cout << "    -fswap                   swap channel ordering for "
                 "negative DM - SIGPROC 2,4 or 8 bit only"
              << std::endl;
    std::cout << "    -debug                   Run in debug mode (output "
                 "filterbank and tseries)"
              << std::endl;
    std::cout << "    -debug_dm                DM Index to dump timeseries in "
                 "debug mode ["
              << p.debug_dm << "]" << std::endl;
    std::cout << "    -min_tscrunch_width num  vary between high quality "
                 "(large value) and high performance (low value)"
              << std::endl;
}

int hd_parse_command_line(int argc, char* argv[], hd_params* params) {
    // TODO: Make this robust to malformed input
    size_t i = 0;
    while (++i < (size_t)argc) {
        if (argv[i] == std::string("-h")) {
            hd_print_usage();
            return -1;
        }
        if (argv[i] == std::string("-v")) {
            params->verbosity = std::max(params->verbosity, 1);
        } else if (argv[i] == std::string("-V")) {
            params->verbosity = std::max(params->verbosity, 2);
        } else if (argv[i] == std::string("-g")) {
            params->verbosity = std::max(params->verbosity, 3);
        } else if (argv[i] == std::string("-G")) {
            params->verbosity = std::max(params->verbosity, 4);
        } else if (argv[i] == std::string("-f")) {
            params->sigproc_file = std::string(argv[++i]);
        } else if (argv[i] == std::string("-yield_cpu")) {
            params->yield_cpu = true;
        } else if (argv[i] == std::string("-nsamps_gulp")) {
            params->nsamps_gulp = atoi(argv[++i]);
        } else if (argv[i] == std::string("-baseline_length")) {
            params->baseline_length = atof(argv[++i]);
        } else if (argv[i] == std::string("-dm")) {
            params->dm_min = atof(argv[++i]);
            params->dm_max = atof(argv[++i]);
        } else if (argv[i] == std::string("-dm_tol")) {
            params->dm_tol = atof(argv[++i]);
        } else if (argv[i] == std::string("-dm_pulse_width")) {
            params->dm_pulse_width = atof(argv[++i]);
        } else if (argv[i] == std::string("-dm_nbits")) {
            params->dm_nbits = atoi(argv[++i]);
        } else if (argv[i] == std::string("-gpu_id")) {
            params->gpu_id = atoi(argv[++i]);
        } else if (argv[i] == std::string("-no_scrunching")) {
            params->use_scrunching = false;
        } else if (argv[i] == std::string("-scrunch_tol")) {
            params->scrunch_tol = atof(argv[++i]);
        } else if (argv[i] == std::string("-rfi_tol")) {
            params->rfi_tol = atof(argv[++i]);
        } else if (argv[i] == std::string("-rfi_no_narrow")) {
            params->rfi_narrow = false;
        } else if (argv[i] == std::string("-rfi_no_broad")) {
            params->rfi_broad = false;
        } else if (argv[i] == std::string("-rfi_boxcar_max")) {
            params->rfi_boxcar_max = atoi(argv[++i]);
        } else if (argv[i] == std::string("-boxcar_max")) {
            params->boxcar_max = atoi(argv[++i]);
        } else if (argv[i] == std::string("-detect_thresh")) {
            params->detect_thresh = atof(argv[++i]);
        } else if (argv[i] == std::string("-beam")) {
            params->beam          = atoi(argv[++i]) - 1;
            params->override_beam = true;
        } else if (argv[i] == std::string("-cand_sep_time")) {
            params->cand_sep_time = atoi(argv[++i]);
        } else if (argv[i] == std::string("-cand_sep_filter")) {
            params->cand_sep_filter = atoi(argv[++i]);
        } else if (argv[i] == std::string("-cand_sep_dm_trial")) {
            params->cand_sep_dm = atoi(argv[++i]);
        } else if (argv[i] == std::string("-cand_rfi_dm_cut")) {
            params->cand_rfi_dm_cut = atof(argv[++i]);
        } else if (argv[i] == std::string("-max_giant_rate")) {
            params->max_giant_rate = atof(argv[++i]);
        } else if (argv[i] == std::string("-output_dir")) {
            params->output_dir = std::string(argv[++i]);
        } else if (argv[i] == std::string("-min_tscrunch_width")) {
            params->min_tscrunch_width = atoi(argv[++i]);
        } else if (argv[i] == std::string("-fswap")) {
            params->fswap = true;
        } else if (argv[i] == std::string("-debug")) {
            params->debug = true;
        } else if (argv[i] == std::string("-debug_dm")) {
            params->debug_dm = atoi(argv[++i]);
        } else if (argv[i] == std::string("-zap_chans")) {
            unsigned int izap = params->num_channel_zaps;
            params->num_channel_zaps++;
            params->channel_zaps = (hd_range_t*)realloc(
                params->channel_zaps,
                sizeof(hd_range_t) * params->num_channel_zaps);
            params->channel_zaps[izap].start = atoi(argv[++i]);
            params->channel_zaps[izap].end   = atoi(argv[++i]);
        } else {
            std::cerr << "WARNING: Unknown parameter '" << argv[i] << "'"
                      << std::endl;
        }
    }

    if (params->sigproc_file.empty()) {
        std::cerr << "ERROR: no input mechanism specified" << std::endl;
        hd_print_usage();
        return -1;
    } else
        return 0;
}
