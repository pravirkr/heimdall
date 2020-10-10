/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/version.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <hd/pipeline.hpp>
#include <hd/clean_filterbank_rfi.hpp>
#include <hd/remove_baseline.hpp>
#include <hd/matched_filter.hpp>
#include <hd/get_rms.hpp>
#include <hd/find_giants.hpp>
#include <hd/label_candidate_clusters.hpp>
#include <hd/merge_candidates.hpp>

#include <data_types/write_time_series.hpp>  // For debugging
#include <data_types/data_source.hpp>
#include <utils/stopwatch.hpp>  // For benchmarking

#include <dedisp/dedisp.hpp>

void start_timer(Stopwatch& timer) { timer.start(); }
void stop_timer(Stopwatch& timer) {
    cudaDeviceSynchronize();
    timer.stop();
}


struct hd_pipeline_t {
    hd_params   params;
    dedisp_plan dedispersion_plan;

    // Memory buffers used during pipeline execution
    std::vector<hd_byte>            h_clean_filterbank;
    thrust::host_vector<hd_byte>    h_dm_series;
    thrust::device_vector<hd_float> d_time_series;
    thrust::device_vector<hd_float> d_filtered_series;
};

hd_error allocate_gpu(const hd_pipeline pl) {
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    int gpu_idx = pl->params.gpu_id;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_idx);

    cudaError_t cerror = cudaSetDevice(gpu_idx);
    if (cerror != cudaSuccess) {
        fmt::print(stderr, "Could not setCudaDevice to {}: {}\n", 
                   gpu_idx, cudaGetErrorString(cerror));
        return throw_cuda_error(cerror);
    }

    if (!pl->params.yield_cpu) {
        if (pl->params.verbosity >= 1) {
            fmt::print("Using GPU {} ({} {}.{}): Setting CPU to spin\n", 
                gpu_idx, prop.name, prop.major, prop.minor);

        }
        cerror = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        if (cerror != cudaSuccess) {
            return throw_cuda_error(cerror);
        }
    } else {
        if (pl->params.verbosity >= 1) {
            fmt::print("Using GPU {} ({} {}.{}): Setting CPU to yield\n", 
                gpu_idx, prop.name, prop.major, prop.minor);
        }
        // Note: This Yield flag doesn't seem to work properly.
        //   The BlockingSync flag does the job, although it may interfere
        //     with GPU/CPU overlapping (not currently used).
        // cerror = cudaSetDeviceFlags(cudaDeviceScheduleYield);
        cerror = cudaSetDeviceFlags(cudaDeviceBlockingSync);
        if (cerror != cudaSuccess) {
            return throw_cuda_error(cerror);
        }
    }

    return HD_NO_ERROR;
}

unsigned int get_filter_index(unsigned int filter_width) {
    // This function finds log2 of the 32-bit power-of-two number v
    unsigned int              v   = filter_width;
    static const unsigned int b[] = {
        0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000};
    register unsigned int r = (v & b[0]) != 0;
    for (int i = 4; i > 0; --i) {
        r |= ((v & b[i]) != 0) << i;
    }
    return r;
}

hd_error hd_create_pipeline(hd_pipeline* pipeline_, hd_params params) {
    *pipeline_ = 0;

    // Note: We use a smart pointer here to automatically clean up after errors
    typedef std::unique_ptr<hd_pipeline_t> smart_pipeline_ptr;
    smart_pipeline_ptr pipeline = smart_pipeline_ptr(new hd_pipeline_t());
    if (!pipeline.get()) {
        return throw_error(HD_MEM_ALLOC_FAILED);
    }

    pipeline->params = params;

    // GPU allocation
    if (params.verbosity >= 2) {
        fmt::print("Allocating GPU...\n");
    }

    hd_error error = allocate_gpu(pipeline.get());
    if (error != HD_NO_ERROR) {
        return throw_error(error);
    }

    if (params.verbosity >= 2) {
        fmt::print("Filterbank header:\n");
        fmt::print("nchans = {}\ndt = {}\nf0 = {}\ndf = {}\n",
                    params.nchans, params.dt, params.f0, params.df);
        fmt::print("Creating dedispersion plan...\n");
    }

    dedisp_error derror;
    derror = dedisp_create_plan(&pipeline->dedispersion_plan,
                                params.nchans,
                                params.dt,
                                params.f0,
                                params.df);
    if (derror != DEDISP_NO_ERROR) {
        return throw_dedisp_error(derror);
    }

    // TODO: Consider loading a pre-generated DM list instead for flexibility
    derror = dedisp_generate_dm_list(pipeline->dedispersion_plan,
                                     pipeline->params.dm_min,
                                     pipeline->params.dm_max,
                                     pipeline->params.dm_pulse_width,
                                     pipeline->params.dm_tol);
    if (derror != DEDISP_NO_ERROR) {
        return throw_dedisp_error(derror);
    }

    if (pipeline->params.use_scrunching) {
        derror = dedisp_enable_adaptive_dt(pipeline->dedispersion_plan,
                                           pipeline->params.dm_pulse_width,
                                           pipeline->params.scrunch_tol);
        if (derror != DEDISP_NO_ERROR) {
            return throw_dedisp_error(derror);
        }
    }

    *pipeline_ = pipeline.release();

    if (params.verbosity >= 2) {
        fmt::print("Initialisation complete. (Using Thrust v{}.{}.{})\n",
                   THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION, 
                   THRUST_SUBMINOR_VERSION);
    }

    return HD_NO_ERROR;
}

hd_error hd_execute(hd_pipeline    pl,
                    const hd_byte* h_filterbank,
                    hd_size        nsamps,
                    hd_size        nbits,
                    hd_size        first_idx,
                    hd_size*       nsamps_processed) {
    hd_error error = HD_NO_ERROR;

    Stopwatch total_timer;
    Stopwatch memory_timer;
    Stopwatch clean_timer;
    Stopwatch dedisp_timer;
    Stopwatch communicate_timer;
    Stopwatch copy_timer;
    Stopwatch baseline_timer;
    Stopwatch normalise_timer;
    Stopwatch filter_timer;
    Stopwatch coinc_timer;
    Stopwatch giants_timer;
    Stopwatch candidates_timer;

    start_timer(total_timer);

    // Note: Filterbank cleaning must be done out-of-place
    hd_size nbytes = nsamps * pl->params.nchans * nbits / 8;
    start_timer(memory_timer);
    pl->h_clean_filterbank.resize(nbytes);
    std::vector<int> h_killmask(pl->params.nchans, 1);
    stop_timer(memory_timer);

    if (pl->params.verbosity >= 2) {
        fmt::print("Cleaning 0-DM filterbank...\n");
    }

    start_timer(clean_timer);
    // Start by cleaning up the filterbank based on the zero-DM time series
    hd_float cleaning_dm = 0.f;
    if( pl->params.debug ) {
        if (pl->params.verbosity >= 2) {
            fmt::print("Writing dirty filterbank to disk...\n");
        }
        write_host_filterbank(&h_filterbank[0],
                              pl->params.nchans, nsamps, nbits,
                              pl->params.dt, pl->params.f0, pl->params.df,
                              "dirty_filterbank.fil");
    }
    // Note: We only clean the narrowest zero-DM signals; otherwise we
    //         start removing real stuff from higher DMs.
    error = clean_filterbank_rfi(pl->dedispersion_plan,
                                 &h_filterbank[0],
                                 nsamps,
                                 nbits,
                                 &pl->h_clean_filterbank[0],
                                 &h_killmask[0],
                                 cleaning_dm,
                                 pl->params.dt,
                                 pl->params.baseline_length,
                                 pl->params.rfi_tol,
                                 pl->params.rfi_broad,
                                 pl->params.rfi_narrow,
                                 pl->params.rfi_boxcar_max);
    if (error != HD_NO_ERROR) {
        return throw_error(error);
    }

    if (pl->params.verbosity >= 2) {
        fmt::print("Applying manual killmasks...\n");
    }

    error = apply_manual_killmasks(pl->dedispersion_plan,
                                   &h_killmask[0],
                                   pl->params.num_channel_zaps,
                                   pl->params.channel_zaps);
    if (error != HD_NO_ERROR) {
        return throw_error(error);
    }

    hd_size good_chan_count =
        thrust::reduce(h_killmask.begin(), h_killmask.end());
    hd_size bad_chan_count = pl->params.nchans - good_chan_count;
    if (pl->params.verbosity >= 2) {
        fmt::print("Bad channel count = {}\n", bad_chan_count);
    }

    // TESTING
    // h_clean_filterbank.assign(h_filterbank, h_filterbank+nbytes);

    stop_timer(clean_timer);

    if( pl->params.debug ) {
        if (pl->params.verbosity >= 2) {
            fmt::print("Writing killmask to disk...\n");
        }
        std::ofstream killfile("killmask.dat");
        for( size_t i=0; i<h_killmask.size(); ++i ) {
            killfile << h_killmask[i] << "\n";
        }
        killfile.close();

        if (pl->params.verbosity >= 2) {
            fmt::print("Writing cleaned filterbank to disk...\n");
        }
        write_host_filterbank(&pl->h_clean_filterbank[0],
                              pl->params.nchans, nsamps, nbits,
                              pl->params.dt, pl->params.f0, pl->params.df,
                              "clean_filterbank.fil");

    }

    if (pl->params.verbosity >= 2) {
        fmt::print("Generating DM list in range ({}, {}) with tol: {}\n",
                    pl->params.dm_min, pl->params.dm_max, pl->params.dm_tol);

    }

    if (pl->params.verbosity >= 3) {
        fmt::print("\tand pulse width: {}\n", pl->params.dm_pulse_width);
        fmt::print("Dedisp params:\n");
        fmt::print("nchans = {}\ndt = {}\nf0 = {}\ndf = {}\n",
                    dedisp_get_channel_count(pl->dedispersion_plan),
                    dedisp_get_dt(pl->dedispersion_plan),
                    dedisp_get_f0(pl->dedispersion_plan),
                    dedisp_get_df(pl->dedispersion_plan));
    }

    hd_size      dm_count = dedisp_get_dm_count(pl->dedispersion_plan);
    const float* dm_list  = dedisp_get_dm_list(pl->dedispersion_plan);

    const dedisp_size* scrunch_factors =
        dedisp_get_dt_factors(pl->dedispersion_plan);

    if (pl->params.verbosity >= 3) {
        fmt::print("DM : Scrunch factor\n");
        for (hd_size i = 0; i < dm_count; ++i) {
            fmt::print("{:<12}{:>5}\n", dm_list[i], scrunch_factors[i]);
        }
    }

    // Set channel killmask for dedispersion
    dedisp_error derror;
    derror = dedisp_set_killmask(pl->dedispersion_plan, &h_killmask[0]);
    if (derror != DEDISP_NO_ERROR) {
        return throw_dedisp_error(derror);
    }

    if (dedisp_get_max_delay(pl->dedispersion_plan) > nsamps) {
        fmt::print(stderr, 
            "Number of samples requested = {}, maximum DM delay = {}\n",
            nsamps, dedisp_get_max_delay(pl->dedispersion_plan));
        return throw_error(HD_TOO_FEW_NSAMPS);
    }

    hd_size nsamps_computed =
        nsamps - dedisp_get_max_delay(pl->dedispersion_plan);
    hd_size series_stride = nsamps_computed;

    // Report the number of samples that will be properly processed
    *nsamps_processed = nsamps_computed - pl->params.boxcar_max;

    if (pl->params.verbosity >= 2) {
        fmt::print(
            "Total DM count: {}, Max delay (smear): {}, Nsamps computed: {}\n",
            dm_count, dedisp_get_max_delay(pl->dedispersion_plan), 
            nsamps_computed);
        fmt::print("Allocating memory for pipeline computations...\n");
    }

    start_timer(memory_timer);

    pl->h_dm_series.resize(series_stride * pl->params.dm_nbits / 8 * dm_count);
    pl->d_time_series.resize(series_stride);
    pl->d_filtered_series.resize(series_stride, 0);

    stop_timer(memory_timer);

    RemoveBaselinePlan          baseline_remover;
    GetRMSPlan                  rms_getter;
    MatchedFilterPlan<hd_float> matched_filter_plan;
    GiantFinder                 giant_finder;

    thrust::device_vector<hd_float> d_giant_peaks;
    thrust::device_vector<hd_size>  d_giant_inds;
    thrust::device_vector<hd_size>  d_giant_begins;
    thrust::device_vector<hd_size>  d_giant_ends;
    thrust::device_vector<hd_size>  d_giant_filter_inds;
    thrust::device_vector<hd_size>  d_giant_dm_inds;
    thrust::device_vector<hd_size>  d_giant_members;

    if (pl->params.verbosity >= 2) {
        fmt::print("Dedispersing for DMs {} to {}...\n",
            dm_list[0], dm_list[dm_count - 1]);
    }

    // Dedisperse
    const dedisp_byte* in         = &pl->h_clean_filterbank[0];
    dedisp_byte*       out        = &pl->h_dm_series[0];
    dedisp_size        in_nbits   = nbits;
    dedisp_size        in_stride  = pl->params.nchans * in_nbits / 8;
    dedisp_size        out_nbits  = pl->params.dm_nbits;
    dedisp_size        out_stride = series_stride * out_nbits / 8;
    unsigned           flags      = 0;
    start_timer(dedisp_timer);
    derror = dedisp_execute_adv(pl->dedispersion_plan,
                                nsamps,
                                in,
                                in_nbits,
                                in_stride,
                                out,
                                out_nbits,
                                out_stride,
                                flags);
    stop_timer(dedisp_timer);
    if (derror != DEDISP_NO_ERROR) {
        return throw_dedisp_error(derror);
    }

    if (pl->params.verbosity >= 2) {
        fmt::print("\tBeginning inner pipeline...\n");
    }


    bool too_many_giants = false;

    // For each DM
    for (hd_size dm_idx = 0; dm_idx < dm_count; ++dm_idx) {
        hd_size  cur_dm_scrunch = scrunch_factors[dm_idx];
        hd_size  cur_nsamps     = nsamps_computed / cur_dm_scrunch;
        hd_float cur_dt         = pl->params.dt * cur_dm_scrunch;

        // Bail if the candidate rate is too high
        if (too_many_giants) {
            break;
        }

        if (pl->params.verbosity >= 3) {
            fmt::print("dm_idx     = {}\n", dm_idx);
            fmt::print("scrunch    = {}\n", scrunch_factors[dm_idx]);
            fmt::print("cur_nsamps = {}\n", cur_nsamps);
            fmt::print("dt0        = {}\n", pl->params.dt);
            fmt::print("cur_dt     = {}\n", cur_dt);
            fmt::print("\tBaselining and normalising...\n");
        }

        hd_float* time_series = thrust::raw_pointer_cast(&pl->d_time_series[0]);

        // Copy the time series to the device and convert to floats
        hd_size offset = dm_idx * series_stride * pl->params.dm_nbits / 8;
        start_timer(copy_timer);
        switch (pl->params.dm_nbits) {
            case 8:
                thrust::copy((unsigned char*)&pl->h_dm_series[offset],
                             (unsigned char*)&pl->h_dm_series[offset] +
                                 cur_nsamps,
                             pl->d_time_series.begin());
                break;
            case 16:
                thrust::copy((unsigned short*)&pl->h_dm_series[offset],
                             (unsigned short*)&pl->h_dm_series[offset] +
                                 cur_nsamps,
                             pl->d_time_series.begin());
                break;
            case 32:
                // Note: 32-bit implies float, not unsigned int
                thrust::copy((float*)&pl->h_dm_series[offset],
                             (float*)&pl->h_dm_series[offset] + cur_nsamps,
                             pl->d_time_series.begin());
                break;
            default:
                return HD_INVALID_NBITS;
        }
        stop_timer(copy_timer);

        if( pl->params.debug && dm_idx == pl->params.debug_dm 
            && first_idx == 0 ) {
            write_device_time_series(time_series, cur_nsamps, cur_dt, 
                                     dm_list[dm_idx], "dedispersed.tim");
        }

        // Remove the baseline
        // -------------------
        start_timer(baseline_timer);
        // Note: Divided by 2 to form a smoothing radius
        hd_size nsamps_smooth =
            hd_size(pl->params.baseline_length / (2 * cur_dt));
        // Crop the smoothing length in case not enough samples
        error = baseline_remover.exec(time_series, cur_nsamps, nsamps_smooth);
        if (error != HD_NO_ERROR) {
            return throw_error(error);
        }
        stop_timer(baseline_timer);

        if( pl->params.debug && dm_idx == pl->params.debug_dm 
            && first_idx == 0 ) {
            write_device_time_series(time_series, cur_nsamps, cur_dt, 
                                     dm_list[dm_idx], "baselined.tim");
        }
        // -------------------

        // Normalise timeseries
        // --------------------
        start_timer(normalise_timer);
        hd_float rms = rms_getter.exec(time_series, cur_nsamps);
        thrust::transform(pl->d_time_series.begin(),
                          pl->d_time_series.end(),
                          thrust::make_constant_iterator(hd_float(1.0) / rms),
                          pl->d_time_series.begin(),
                          thrust::multiplies<hd_float>());
        stop_timer(normalise_timer);

        if( pl->params.debug && dm_idx == pl->params.debug_dm 
            && first_idx == 0 ) {
            write_device_time_series(time_series, cur_nsamps, cur_dt, 
                                     dm_list[dm_idx], "normalised.tim");
        }
        // ---------

        // Prepare the boxcar filters
        // --------------------------
        start_timer(filter_timer);
        // We can't process the first and last max-filter-width/2 samples
        hd_size rel_boxcar_max      = pl->params.boxcar_max / cur_dm_scrunch;
        hd_size max_nsamps_filtered = cur_nsamps + 1 - rel_boxcar_max;
        // This is the relative offset into the time series of the filtered data
        hd_size cur_filtered_offset = rel_boxcar_max / 2;

        // Create and prepare matched filtering operations
        // Note: Filter width is relative to the current time resolution
        matched_filter_plan.prep(time_series, cur_nsamps, rel_boxcar_max);
        stop_timer(filter_timer);
        // --------------------------

        hd_float* filtered_series =
            thrust::raw_pointer_cast(&pl->d_filtered_series[0]);

        // Note: Filtering is done using a combination of tscrunching and
        //         'proper' boxcar convolution. The parameter min_tscrunch_width
        //         indicates how much of each to do. Raising min_tscrunch_width
        //         increases sensitivity but decreases performance and vice
        //         versa.

        // For each boxcar filter
        // Note: We cannot detect pulse widths < current time resolution
        for (hd_size filter_width = cur_dm_scrunch;
             filter_width <= pl->params.boxcar_max;
             filter_width *= 2) {
            hd_size rel_filter_width = filter_width / cur_dm_scrunch;
            hd_size filter_idx       = get_filter_index(filter_width);

            if (pl->params.verbosity >= 3) {
                fmt::print(
                    "Filtering each tseries at width of {}, filter_idx={}\n",
                    filter_width, filter_idx);
            }

            // Note: Filter width is relative to the current time resolution
            hd_size rel_min_tscrunch_width = std::max(
                pl->params.min_tscrunch_width / cur_dm_scrunch, hd_size(1));
            hd_size rel_tscrunch_width = std::max(
                2 * rel_filter_width / rel_min_tscrunch_width, hd_size(1));
            // Filter width relative to cur_dm_scrunch AND tscrunch
            hd_size rel_rel_filter_width =
                rel_filter_width / rel_tscrunch_width;

            start_timer(filter_timer);
            error = matched_filter_plan.exec(
                filtered_series, rel_filter_width, rel_tscrunch_width);
            if (error != HD_NO_ERROR) {
                return throw_error(error);
            }
            // Divide and round up
            hd_size cur_nsamps_filtered =
                ((max_nsamps_filtered - 1) / rel_tscrunch_width + 1);
            hd_size cur_scrunch = cur_dm_scrunch * rel_tscrunch_width;

            // Normalise the filtered time series (RMS ~ sqrt(time))
            // TODO: Avoid/hide the ugly thrust code?
            //         Consider making it a method of MatchedFilterPlan
            /*
            thrust::constant_iterator<hd_float>
              norm_val_iter(1.0 / sqrt((hd_float)rel_filter_width));
            thrust::transform(thrust::device_ptr<hd_float>(filtered_series),
                              thrust::device_ptr<hd_float>(filtered_series)
                              + cur_nsamps_filtered,
                              norm_val_iter,
                              thrust::device_ptr<hd_float>(filtered_series),
                              thrust::multiplies<hd_float>());
            */
            // TESTING Proper normalisation
            hd_float rms =
                rms_getter.exec(filtered_series, cur_nsamps_filtered);
            thrust::transform(
                thrust::device_ptr<hd_float>(filtered_series),
                thrust::device_ptr<hd_float>(filtered_series) +
                    cur_nsamps_filtered,
                thrust::make_constant_iterator(hd_float(1.0) / rms),
                thrust::device_ptr<hd_float>(filtered_series),
                thrust::multiplies<hd_float>());

            stop_timer(filter_timer);

            if( pl->params.debug && dm_idx == pl->params.debug_dm 
                && first_idx == 0 && filter_width == 8 ) {
                write_device_time_series(filtered_series, cur_nsamps_filtered,
                                         cur_dt, dm_list[dm_idx], "filtered.tim");
            }
            // --------------------------

            hd_size prev_giant_count = d_giant_peaks.size();

            start_timer(giants_timer);

            if (pl->params.verbosity >= 3) {
                fmt::print("Finding giants...\n");
                fmt::print(stderr, 
                    "pl->params.cand_sep_time={} rel_rel_filter_width={}\n",
                    pl->params.cand_sep_time, rel_rel_filter_width);
            }

            error = giant_finder.exec(filtered_series,
                                      cur_nsamps_filtered,
                                      pl->params.detect_thresh,
                                      // pl->params.cand_sep_time,
                                      // Note: This was MB's recommendation
                                      pl->params.cand_sep_time *
                                          rel_rel_filter_width,
                                      d_giant_peaks,
                                      d_giant_inds,
                                      d_giant_begins,
                                      d_giant_ends);

            if (error != HD_NO_ERROR) {
                return throw_error(error);
            }

            hd_size rel_cur_filtered_offset =
                (cur_filtered_offset / rel_tscrunch_width);

            using namespace thrust::placeholders;
            thrust::transform(d_giant_inds.begin() + prev_giant_count,
                              d_giant_inds.end(),
                              d_giant_inds.begin() + prev_giant_count,
                              /*first_idx +*/ (_1 + rel_cur_filtered_offset) *
                                  cur_scrunch);
            thrust::transform(d_giant_begins.begin() + prev_giant_count,
                              d_giant_begins.end(),
                              d_giant_begins.begin() + prev_giant_count,
                              /*first_idx +*/ (_1 + rel_cur_filtered_offset) *
                                  cur_scrunch);
            thrust::transform(d_giant_ends.begin() + prev_giant_count,
                              d_giant_ends.end(),
                              d_giant_ends.begin() + prev_giant_count,
                              /*first_idx +*/ (_1 + rel_cur_filtered_offset) *
                                  cur_scrunch);

            d_giant_filter_inds.resize(d_giant_peaks.size(), filter_idx);
            d_giant_dm_inds.resize(d_giant_peaks.size(), dm_idx);
            // Note: This could be used to track total member samples if desired
            d_giant_members.resize(d_giant_peaks.size(), 1);

            stop_timer(giants_timer);

            // Bail if the candidate rate is too high
            hd_size  total_giant_count = d_giant_peaks.size();
            hd_float data_length_mins  = nsamps * pl->params.dt / 60.0;
            if (pl->params.max_giant_rate &&
                (total_giant_count / data_length_mins >
                 pl->params.max_giant_rate)) {
                too_many_giants = true;
                float searched  = ((float)dm_idx * 100) / (float)dm_count;
                fmt::print("WARNING: exceeded max giants/min, DM [{}], space searched {}%\n",
                    dm_list[dm_idx], searched);
                break;
            }

        }  // End of filter width loop
    }      // End of DM loop

    hd_size giant_count = d_giant_peaks.size();
    if (pl->params.verbosity >= 2) {
        fmt::print("Giant count = {}\n", giant_count);
    }

    start_timer(candidates_timer);

    thrust::host_vector<hd_float> h_group_peaks;
    thrust::host_vector<hd_size>  h_group_inds;
    thrust::host_vector<hd_size>  h_group_begins;
    thrust::host_vector<hd_size>  h_group_ends;
    thrust::host_vector<hd_size>  h_group_filter_inds;
    thrust::host_vector<hd_size>  h_group_dm_inds;
    thrust::host_vector<hd_size>  h_group_members;
    thrust::host_vector<hd_float> h_group_dms;

    thrust::device_vector<hd_size> d_giant_labels(giant_count);
    hd_size* d_giant_labels_ptr = thrust::raw_pointer_cast(&d_giant_labels[0]);

    RawCandidates d_giants;
    d_giants.peaks       = thrust::raw_pointer_cast(&d_giant_peaks[0]);
    d_giants.inds        = thrust::raw_pointer_cast(&d_giant_inds[0]);
    d_giants.begins      = thrust::raw_pointer_cast(&d_giant_begins[0]);
    d_giants.ends        = thrust::raw_pointer_cast(&d_giant_ends[0]);
    d_giants.filter_inds = thrust::raw_pointer_cast(&d_giant_filter_inds[0]);
    d_giants.dm_inds     = thrust::raw_pointer_cast(&d_giant_dm_inds[0]);
    d_giants.members     = thrust::raw_pointer_cast(&d_giant_members[0]);

    hd_size filter_count = get_filter_index(pl->params.boxcar_max) + 1;

    if (pl->params.verbosity >= 2) {
        fmt::print("Grouping coincident candidates...\n");
    }

    ConstRawCandidates* const_d_giants = (ConstRawCandidates*)&d_giants;

    hd_size label_count;
    error = label_candidate_clusters(giant_count,
                                     *const_d_giants,
                                     pl->params.cand_sep_time,
                                     pl->params.cand_sep_filter,
                                     pl->params.cand_sep_dm,
                                     d_giant_labels_ptr,
                                     &label_count);
    if (error != HD_NO_ERROR) {
        return throw_error(error);
    }

    hd_size group_count = label_count;
    if (pl->params.verbosity >= 2) {
        fmt::print("Candidate count = {}\n", group_count);
    }

    thrust::device_vector<hd_float> d_group_peaks(group_count);
    thrust::device_vector<hd_size>  d_group_inds(group_count);
    thrust::device_vector<hd_size>  d_group_begins(group_count);
    thrust::device_vector<hd_size>  d_group_ends(group_count);
    thrust::device_vector<hd_size>  d_group_filter_inds(group_count);
    thrust::device_vector<hd_size>  d_group_dm_inds(group_count);
    thrust::device_vector<hd_size>  d_group_members(group_count);
    thrust::device_vector<hd_float> d_group_dms(group_count);

    RawCandidates d_groups;
    d_groups.peaks       = thrust::raw_pointer_cast(&d_group_peaks[0]);
    d_groups.inds        = thrust::raw_pointer_cast(&d_group_inds[0]);
    d_groups.begins      = thrust::raw_pointer_cast(&d_group_begins[0]);
    d_groups.ends        = thrust::raw_pointer_cast(&d_group_ends[0]);
    d_groups.filter_inds = thrust::raw_pointer_cast(&d_group_filter_inds[0]);
    d_groups.dm_inds     = thrust::raw_pointer_cast(&d_group_dm_inds[0]);
    d_groups.members     = thrust::raw_pointer_cast(&d_group_members[0]);

    merge_candidates(
        giant_count, d_giant_labels_ptr, *const_d_giants, d_groups);

    // Look up the actual DM of each group
    thrust::device_vector<hd_float> d_dm_list(dm_list, dm_list + dm_count);
    thrust::gather(d_group_dm_inds.begin(),
                   d_group_dm_inds.end(),
                   d_dm_list.begin(),
                   d_group_dms.begin());

    // Device to host transfer of candidates
    h_group_peaks       = d_group_peaks;
    h_group_inds        = d_group_inds;
    h_group_begins      = d_group_begins;
    h_group_ends        = d_group_ends;
    h_group_filter_inds = d_group_filter_inds;
    h_group_dm_inds     = d_group_dm_inds;
    h_group_members     = d_group_members;
    h_group_dms         = d_group_dms;

    if (pl->params.verbosity >= 2) {
        fmt::print("Writing output candidates, utc_start = {}\n",
            pl->params.utc_start);
    }

    char buffer[128];
    std::time_t now = pl->params.utc_start +
                 (std::time_t)(first_idx / pl->params.spectra_per_second);
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H:%M:%S", std::gmtime(&now));

    if (pl->params.verbosity >= 2){
        fmt::print("Output timestamp: {}\n", buffer);
    }

    std::string filename = fmt::format("{}/{}_{:02d}.cand",
        pl->params.output_dir, buffer, pl->params.beam + 1);
    if (pl->params.verbosity >= 2){
        fmt::print("Dumping {} candidates to {}\n", h_group_peaks.size(), 
            filename);
    }

    std::ofstream cand_file(filename);
    if (cand_file.good()) {
        for (hd_size i = 0; i < h_group_peaks.size(); ++i) {
            hd_size samp_idx = first_idx + h_group_inds[i];
            fmt::print(cand_file, 
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t\n",
                h_group_peaks[i], samp_idx, samp_idx * pl->params.dt,
                h_group_filter_inds[i], h_group_dm_inds[i],
                h_group_dms[i], h_group_members[i], 
                first_idx + h_group_begins[i], first_idx + h_group_ends[i]);
        }
    } else{
        fmt::print("Skipping dump due to bad file open on {}\n", filename);
    }
    cand_file.close();

    stop_timer(candidates_timer);

    stop_timer(total_timer);

    if (pl->params.verbosity >= 1) {
        fmt::print("\n");
        fmt::print("{:<25}: {}\n", "Mem alloc time", memory_timer.getTime());
        fmt::print("{:<25}: {}\n", "0-DM cleaning time", clean_timer.getTime());
        fmt::print("{:<25}: {}\n", "Dedispersion time", dedisp_timer.getTime());
        fmt::print("{:<25}: {}\n", "Copy time", copy_timer.getTime());
        fmt::print("{:<25}: {}\n", "Baselining time", baseline_timer.getTime());
        fmt::print("{:<25}: {}\n", "Normalisation time", normalise_timer.getTime());
        fmt::print("{:<25}: {}\n", "Filtering time", filter_timer.getTime());
        fmt::print("{:<25}: {}\n", "Find giant time", giants_timer.getTime());
        fmt::print("{:<25}: {}\n", "Process candidates time", candidates_timer.getTime());
        fmt::print("{:<25}: {}\n", "Total time", total_timer.getTime());
    }

    hd_float time_sum  = (memory_timer.getTime() + clean_timer.getTime() +
                         dedisp_timer.getTime() + copy_timer.getTime() +
                         baseline_timer.getTime() + normalise_timer.getTime() +
                         filter_timer.getTime() + giants_timer.getTime() +
                         candidates_timer.getTime());
    hd_float misc_time = total_timer.getTime() - time_sum;

    if (too_many_giants) {
        return HD_TOO_MANY_EVENTS;
    } else {
        return HD_NO_ERROR;
    }
}

void hd_destroy_pipeline(hd_pipeline pipeline) {
    if (pipeline->params.verbosity >= 2) {
        fmt::print("Deleting pipeline object...\n");
    }

    dedisp_destroy_plan(pipeline->dedispersion_plan);

    // Note: This assumes memory owned by pipeline cleans itself up
    if (pipeline) {
        delete pipeline;
    }
}
