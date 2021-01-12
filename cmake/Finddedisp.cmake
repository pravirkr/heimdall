# - Try to find dedisp
# Variables used by this module:
#  dedisp_ROOT_DIR     - dedisp root directory
# Once done this will define
#  dedisp_FOUND - System has dedisp
#  dedisp_INCLUDE_DIRS - The dedisp include directories
#  dedisp_LIBRARIES - The libraries needed to use dedisp

find_path(dedisp_INCLUDE_DIR dedisp/dedisp.hpp dedisp/DedispPlan.hpp
    HINTS ${dedisp_ROOT_DIR}
    PATH_SUFFIXES include
)

find_library(dedisp_LIBRARY NAMES dedisp libdedisp
    HINTS ${dedisp_ROOT_DIR}
    PATH_SUFFIXES lib build build/src
)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set DEDISP_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(dedisp DEFAULT_MSG
    dedisp_LIBRARY dedisp_INCLUDE_DIR
)

mark_as_advanced(dedisp_INCLUDE_DIR dedisp_LIBRARY )

set(dedisp_LIBRARIES ${dedisp_LIBRARY} )
set(dedisp_INCLUDE_DIRS ${dedisp_INCLUDE_DIR} )
