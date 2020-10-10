/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <hd/params.hpp>

void hd_set_default_params(hd_params* params) {
    // Default parameters
    params->gpu_id          = 0;
    params->verbosity       = 0;
    params->yield_cpu       = false;
    params->fswap           = false;
    params->debug           = false;
    params->debug_dm        = 0;
    params->nsamps_gulp     = 262144;
    params->output_dir      = ".";
    params->baseline_length = 2.0;
    params->beam            = 0;
    params->override_beam   = false;
    params->nchans          = 1024;
    params->dt              = 64e-6;
    params->f0              = 1581.804688;
    params->df              = -.390625;
    params->utc_start       = 0;
    params->dm_min          = 0.0;
    params->dm_max          = 1000.0;
    params->dm_tol          = 1.25;
    params->dm_pulse_width  = 40;  // in microseconds
    params->dm_nbits        = 32;
    params->use_scrunching  = true;
    params->scrunch_tol     = 1.15;
    params->rfi_tol         = 5.0;
    params->rfi_narrow      = true;
    params->rfi_broad       = true;
    params->rfi_boxcar_max  = 1;
    params->boxcar_max      = 4096;
    params->detect_thresh   = 6.0;
    // TODO: This still needs tuning!
    params->max_giant_rate  = 0;  // per minute, 0 == no limit
    params->cand_rfi_dm_cut = 1.5;
    params->cand_sep_dm     = 200;  // Note: trials, not actual DM
    params->cand_sep_time   = 3;
    // Note: These have very little effect on the candidates, but could be
    // important to capture (*rare*) coincident events.
    params->cand_sep_filter    = 3;  // Note: filter numbers, not actual width
    params->num_channel_zaps   = 0;
    params->channel_zaps       = NULL;
    params->min_tscrunch_width = 4096;
    params->spectra_per_second = 0;
}
