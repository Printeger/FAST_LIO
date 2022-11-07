#ifndef POINT_TYPE_H_
#define POINT_TYPE_H_

#define PCL_NO_PRECOMPILE
// #include <pcl/memory.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

//  x y z timestamp intensity flags elongation scan_id scan_idx
struct EIGEN_ALIGN16 PointXYZTIFES  // enforce SSE padding for correct memory alignment
{
    PCL_ADD_POINT4D;  // preferred way of adding a XYZ+padding
    double timestamp;
    std::uint16_t intensity;
    std::uint8_t flags;
    std::uint8_t elongation;
    std::uint16_t scan_id;
    std::uint16_t scan_idx;
    // PCL_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
// here we assume a XYZ + "test" (as fields)
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZTIFES,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (double, timestamp, timestamp)
                                  (std::uint16_t, intensity, intensity)
                                  (std::uint8_t, flags, flags)
                                  (std::uint8_t, elongation, elongation)
                                  (std::uint16_t, scan_id, scan_id)
                                  (std::uint16_t, scan_idx,scan_idx))
#endif
