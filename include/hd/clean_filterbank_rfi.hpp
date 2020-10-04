/***************************************************************************
 *  *
 *   *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *    *   Licensed under the Academic Free License version 2.1
 *     *
 *      ***************************************************************************/

#pragma once

#include <dedisp.h>
#include <hd/error.hpp>
#include <hd/pipeline_types.hpp>

hd_error clean_filterbank_rfi(dedisp_plan    plan,
                              const hd_byte* h_in,
                              hd_size        nsamps,
                              hd_size        nbits,
                              hd_byte*       h_out,
                              int*           h_killmask,
                              hd_float       dm,
                              hd_float       dt,
                              hd_float       baseline_length,
                              hd_float       rfi_tol,
                              bool           rfi_broad,
                              bool           rfi_narrow,
                              hd_size        boxcar_max);

hd_error apply_manual_killmasks(dedisp_plan  main_plan,
                                int*         h_killmask,
                                unsigned int num_channel_zaps,
                                hd_range_t*  channel_zaps);
