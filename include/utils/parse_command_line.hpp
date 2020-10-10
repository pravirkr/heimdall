/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <string>
#include <cstdlib>
#include <fmt/format.h>
#include <hd/params.hpp>
#include <hd/default_params.hpp>

void hd_print_usage() {
    hd_params p;
    hd_set_default_params(&p);

    fmt::print("Usage: heimdall [options]\n");
    fmt::print("\t{:<25} {}\n",
               "-f filename",
               "process specified SIGPROC filterbank file");
    fmt::print("\t{:<25} {}\n", "-vVgG", "increase verbosity level");
    fmt::print(
        "\t{:<25} {}\n", "-yield_cpu", "yield CPU during GPU operations");
    fmt::print("\t{:<25} {}\n", "-gpu_id ID", "run on specified GPU");
    fmt::print("\t{:<25} {} [{}]\n",
               "-nsamps_gulp num",
               "number of samples to be read at a time",
               p.nsamps_gulp);
    fmt::print("\t{:<25} {} [{}]\n",
               "-baseline_length num",
               "number of seconds over which to smooth the baseline",
               p.baseline_length);
    fmt::print("\t{:<25} {}\n", "-beam ##", "over-ride beam number");
    fmt::print("\t{:<25} {}\n",
               "-output_dir path",
               "create all output files in the specified path");
    fmt::print("\t{:<25} {}\n", "-dm min max", "min and max DM");
    fmt::print("\t{:<25} {} [{}]\n",
               "-dm_tol num",
               "SNR loss tolerance between each DM trial",
               p.dm_tol);
    fmt::print("\t{:<25} {}\n",
               "-zap_chans start end",
               "zap all channels between start and end channels inclusive");
    fmt::print("\t{:<25} {}\n",
               "-max_giant_rate nevents",
               "limit the maximum number of individual detections per minute "
               "to nevents");
    fmt::print("\t{:<25} {}\n",
               "-dm_pulse_width num",
               "expected intrinsic width of the pulse signal in microseconds");
    fmt::print("\t{:<25} {} [{}]\n",
               "-dm_nbits num",
               "number of bits per sample in dedispersed time series",
               p.dm_nbits);
    fmt::print("\t{:<25} {}\n",
               "-no_scrunching",
               "don't use an adaptive time scrunching during dedispersion");
    fmt::print("\t{:<25} {} [{}]\n",
               "-scrunching_tol num",
               "smear tolerance factor for time scrunching",
               p.scrunch_tol);
    fmt::print("\t{:<25} {} [{}]\n",
               "-rfi_tol num",
               "RFI exicision threshold limits",
               p.rfi_tol);
    fmt::print(
        "\t{:<25} {}\n", "-rfi_no_narrow", "disable narrow band RFI excision");
    fmt::print("\t{:<25} {}\n", "-rfi_no_broad", "disable 0-DM RFI excision");
    fmt::print("\t{:<25} {} [{}]\n",
               "-rfi_boxcar_max num",
               "maximum boxcar width in samples while 0-DM RFI excision",
               p.rfi_boxcar_max);
    fmt::print("\t{:<25} {} [{}]\n",
               "-boxcar_max num",
               "maximum boxcar width in samples",
               p.boxcar_max);
    fmt::print("\t{:<25} {} [{}]\n",
               "-detect_thresh num",
               "Detection threshold (units of std. dev.)",
               p.detect_thresh);
    fmt::print("\t{:<25} {} [{}]\n",
               "-cand_sep_time num",
               "Min separation between candidates (in samples)",
               p.cand_sep_time);
    fmt::print("\t{:<25} {} [{}]\n",
               "-cand_sep_filter num",
               "Min separation between candidates (in filters)",
               p.cand_sep_filter);
    fmt::print("\t{:<25} {} [{}]\n",
               "-cand_sep_dm_trial num",
               "Min separation between candidates (in DM trials)",
               p.cand_sep_dm);
    fmt::print("\t{:<25} {} [{}]\n",
               "-cand_rfi_dm_cut num",
               "Minimum DM for valid candidate",
               p.cand_rfi_dm_cut);
    fmt::print("\t{:<25} {}\n",
               "-fswap",
               "swap channel ordering for negative DM - 2,4 or 8 bit only");
    fmt::print("\t{:<25} {}\n",
               "-debug",
               "Run in debug mode (output filterbank and tseries)");
    fmt::print("\t{:<25} {} [{}]\n",
               "-debug_dm",
               "DM Index to dump timeseries in debug mode",
               p.debug_dm);
    fmt::print("\t{:<25} {}\n",
               "-min_tscrunch_width num",
               "vary between high quality (large value) and high performance "
               "(low value)");
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
            fmt::print(stderr, "WARNING: Unknown parameter '{}'\n", argv[i]);
        }
    }

    if (params->sigproc_file.empty()) {
        fmt::print(stderr, "ERROR: no input mechanism specified\n");
        hd_print_usage();
        return -1;
    } else
        return 0;
}
