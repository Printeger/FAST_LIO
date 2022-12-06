#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
// #include <livox_ros_driver/CustomMsg.h>
#include "../../devel/include/fast_lio/inno_point.h"
#include "../../devel/include/fast_lio/inno_points.h"
#include "inno_pcl_point.h"

using namespace std;

#define PCL_NO_PRECOMPILE

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
// typedef PointXYZTIFES PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE
{
    AVIA = 1,
    VELO16,
    OUST64,
    INNO
}; //{1, 2, 3}
enum TIME_UNIT
{
    SEC = 0,
    MS = 1,
    US = 2,
    NS = 3
};
enum Feature
{
    Nor,
    Poss_Plane,
    Real_Plane,
    Edge_Jump,
    Edge_Plane,
    Wire,
    ZeroPoint
};
enum Surround
{
    Prev,
    Next
};
enum E_jump
{
    Nr_nor,
    Nr_zero,
    Nr_180,
    Nr_inf,
    Nr_blind
};

// orgtype类：用于存储激光雷达点的一些其他属性
struct orgtype
{
    double range; // 点云在xy平面离雷达中心的距离
    double dista; // 当前点与后一个点之间的距离
    double angle[2];
    double intersect;
    E_jump edj[2];
    Feature ftype;
    orgtype()
    {
        range = 0;
        edj[Prev] = Nr_nor;
        edj[Next] = Nr_nor;
        ftype = Nor;
        intersect = 2;
    }
};

namespace velodyne_ros
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        float time;
        uint16_t ring;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time, time)(uint16_t, ring, ring))

// namespace inno_ros
// {
struct EIGEN_ALIGN16 PointINNO
{
    PCL_ADD_POINT4D;
    PCL_ADD_NORMAL4D;
    union
    {
        struct
        {
            double timestamp;
            float intensity;
            float curvature;
            std::uint8_t flags;
            std::uint8_t elongation;
            std::uint16_t scan_id;
            std::uint16_t scan_idx;
        };
        float data_c[4];
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
// } // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(PointINNO,
                                  (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
                                      float, normal_y, normal_y)(float, normal_z, normal_z)(double, timestamp, timestamp)(uint16_t, intensity, intensity)(uint8_t, flags, flags)(uint8_t, elongation, elongation)(uint16_t, scan_id, scan_id)(uint16_t, scan_idx, scan_idx))

class Preprocess
{
public:
    //   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Preprocess();
    ~Preprocess();

    // void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
    void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
    // void process(const fast_lio::inno_points::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
    void set(bool feat_en, int lid_type, double bld, int pfilt_num);

    // sensor_msgs::PointCloud2::ConstPtr pointcloud;
    PointCloudXYZI pl_full, pl_corn, pl_surf;
    PointCloudXYZI pl_buff[128];      // maximum 128 line lidar
    vector<orgtype> typess[128];      // maximum 128 line lidar
    PointCloudXYZI pl_buff_inno[300]; // maximum 128 line lidar
    vector<orgtype> typess_inno[300]; // maximum 128 line lidar
    float time_unit_scale;
    int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;
    double blind; // 最小距离阈值(盲区)
    bool feature_enabled, given_offset_time;
    ros::Publisher pub_full, pub_surf, pub_corn;

private:
    // void inno_handler(const fast_lio::inno_points::ConstPtr &msg);
    void inno_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
    void pub_func(PointCloudXYZI &pl, const ros::Time &ct);
    int plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
    bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
    bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);

    int group_size;
    double disA, disB, inf_bound;
    double limit_maxmid, limit_midmin, limit_maxmin;
    double p2l_ratio;
    double jump_up_limit, jump_down_limit;
    double cos160;
    double edgea, edgeb;
    double smallp_intersect, smallp_ratio;
    double vx, vy, vz;
};
