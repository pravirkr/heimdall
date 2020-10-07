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

struct RemoveBaselinePlan_impl;

struct RemoveBaselinePlan {
    RemoveBaselinePlan();
    hd_error exec(hd_float* d_data, hd_size count, hd_size smooth_radius);

private:
    std::shared_ptr<RemoveBaselinePlan_impl> m_impl;
};
