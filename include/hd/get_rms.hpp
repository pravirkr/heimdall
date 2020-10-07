/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <memory>

#include <hd/error.hpp>
#include <hd/pipeline_types.hpp>

struct GetRMSPlan_impl;

struct GetRMSPlan {
    GetRMSPlan();
    hd_float exec(hd_float* d_data, hd_size count);

private:
    std::shared_ptr<GetRMSPlan_impl> m_impl;
};

// Convenience functions for one-off calls
hd_float get_rms(hd_float* d_data, hd_size count);
hd_error normalise(hd_float* d_data, hd_size count);
