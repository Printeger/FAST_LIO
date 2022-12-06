#ifndef FEATURE_EXTRACT_H_
#define FEATURE_EXTRACT_H_

#define PCL_NO_PRECOMPILE

#include <chrono>
// PCL
#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/normal_space.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/point_types.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/impl/mls.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
// EIGEN
#include <Eigen/Dense>

// #include "data_proc.h"
#include "inno_pcl_point.h"
#include "preprocess.h"

#define max_(a, b) (((a) > (b)) ? (a) : (b))
#define min_(a, b) (((a) < (b)) ? (a) : (b))
// MULLS PCA
struct eigenvalue_t // Eigen Value ,lamada1 > lamada2 > lamada3;
{
    double lamada1;
    double lamada2;
    double lamada3;
};
struct eigenvector_t // the eigen vector corresponding to the eigen value
{
    Eigen::Vector3f principalDirection;
    Eigen::Vector3f middleDirection;
    Eigen::Vector3f normalDirection;
};
struct pca_feature_t
{
    eigenvalue_t values;
    eigenvector_t vectors;
    double curvature;
    double linear;
    double planar;
    double spherical;
    double linear_2;
    double planar_2;
    double spherical_2;
    double normal_diff_ang_deg;
    pcl::PointNormal pt;
    int ptId;
    int pt_num = 0;
    std::vector<int> neighbor_indices;
    std::vector<bool> close_to_query_point;
};
// regular bounding box whose edges are parallel to x,y,z axises
struct bounds_t
{
    double min_x;
    double min_y;
    double min_z;
    double max_x;
    double max_y;
    double max_z;
    int type;

    bounds_t()
    {
        min_x = min_y = min_z = max_x = max_y = max_z = 0.0;
    }
    void inf_x()
    {
        min_x = -DBL_MAX;
        max_x = DBL_MAX;
    }
    void inf_y()
    {
        min_y = -DBL_MAX;
        max_y = DBL_MAX;
    }
    void inf_z()
    {
        min_z = -DBL_MAX;
        max_z = DBL_MAX;
    }
    void inf_xyz()
    {
        inf_x();
        inf_y();
        inf_z();
    }
};

struct centerpoint_t
{
    double x;
    double y;
    double z;
    centerpoint_t(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z)
    {
    }
};

struct grid_t
{
    std::vector<int> point_id;
    float min_z;
    float max_z;
    float delta_z;
    float min_z_x; // X of Lowest Point in the Voxel;
    float min_z_y; // Y of Lowest Point in the Voxel;
    float min_z_outlier_thre;
    float neighbor_min_z;
    int pts_count;
    int reliable_neighbor_grid_num;
    float mean_z;
    float dist2station;

    grid_t()
    {
        min_z = min_z_x = min_z_y = neighbor_min_z = mean_z = 0.f;
        pts_count = 0;
        reliable_neighbor_grid_num = 0;
        delta_z = 0.0;
        dist2station = 0.001;
        min_z_outlier_thre = -FLT_MAX;
    }
};

struct idpair_t
{
    idpair_t() : idx(0), voxel_idx(0)
    {
    }
    unsigned long long voxel_idx;
    int idx;
    bool operator<(const idpair_t &pair)
    {
        return voxel_idx < pair.voxel_idx;
    }
};

class PrincipleComponentAnalysis
{
public:
    /**
     * \brief Estimate the normals of the input Point Cloud by PCL speeding up with OpenMP
     * \param[in] in_cloud is the input Point Cloud Pointer
     * \param[in] radius is the neighborhood search radius (m) for KD Tree
     * \param[out] normals is the normal of all the points from the Point Cloud
     */
    bool get_normal_pcar(pcl::PointCloud<PointINNO>::Ptr in_cloud, float radius,
                         pcl::PointCloud<pcl::Normal>::Ptr &normals)
    {
        // Create the normal estimation class, and pass the input dataset to it;
        pcl::NormalEstimationOMP<PointINNO, pcl::Normal> ne;
        // ne.setNumberOfThreads(omp_get_max_threads()); // More threads sometimes would not speed up the procedure
        ne.setInputCloud(in_cloud);
        // Create an empty kd-tree representation, and pass it to the normal estimation object;
        pcl::search::KdTree<PointINNO>::Ptr tree(new pcl::search::KdTree<PointINNO>());
        ne.setSearchMethod(tree);
        // Use all neighbors in a sphere of radius;
        ne.setRadiusSearch(radius);
        // Compute the normal
        ne.compute(*normals);
        check_normal(normals);
        return true;
    }
    bool get_normal_pcak(pcl::PointCloud<PointINNO>::Ptr in_cloud, int K, pcl::PointCloud<pcl::Normal>::Ptr &normals)
    {
        // Create the normal estimation class, and pass the input dataset to it;
        pcl::NormalEstimationOMP<PointINNO, pcl::Normal> ne;
        // ne.setNumberOfThreads(omp_get_max_threads()); // More threads sometimes would not speed up the procedure
        ne.setInputCloud(in_cloud);
        // Create an empty kd-tree representation, and pass it to the normal estimation object;
        pcl::search::KdTree<PointINNO>::Ptr tree(new pcl::search::KdTree<PointINNO>());
        ne.setSearchMethod(tree);
        // Use all neighbors in a sphere of radius;
        ne.setKSearch(K);
        // Compute the normal
        ne.compute(*normals);
        check_normal(normals);
        return true;
    }

    bool get_pc_pca_feature(pcl::PointCloud<PointINNO>::Ptr in_cloud, std::vector<pca_feature_t> &features,
                            pcl::KdTreeFLANN<PointINNO>::Ptr &tree, float radius, int nearest_k, int min_k = 1,
                            int pca_down_rate = 1, bool distance_adaptive_on = false, float unit_dist = 35.0)
    {
        // std::cout << "\n" << "[" << in_cloud->points.size() << "] points used for PCA, pca
        // down rate is [" << pca_down_rate << "]";
        features.resize(in_cloud->points.size());
        // TicToc t_get_pca;
        double max_x = 0.0;
        double max_y = 0.0;
        for (int i = 0; i < in_cloud->points.size(); i += pca_down_rate) // faster way
        {
            // if (i % pca_down_rate == 0) {//this way is much slower
            std::vector<int> search_indices_used; // points would be stored in sequence (from the
                                                  // closest point to the farthest point within
                                                  // the neighborhood)
            std::vector<int> search_indices;      // point index vector
            std::vector<float> squared_distances; // distance vector

            float neighborhood_r = radius;
            int neighborhood_k = nearest_k;

            if (distance_adaptive_on)
            {
                std::cout << "distance_adaptive_on" << std::endl;
                double dist = std::sqrt(in_cloud->points[i].x * in_cloud->points[i].x +
                                        in_cloud->points[i].y * in_cloud->points[i].y +
                                        in_cloud->points[i].z * in_cloud->points[i].z);
                if (dist > unit_dist)
                {
                    neighborhood_r = std::sqrt(dist / unit_dist) * radius;
                    // neighborhood_k = (int)(unit_dist / dist * nearest_k));
                }
            }
            // nearest_k=0 --> the knn is disabled, only the rnn is used
            // 0.001~0.002ms
            //点周围半径内搜索
            tree->radiusSearch(i, neighborhood_r, search_indices, squared_distances, neighborhood_k);
            features[i].pt.x = in_cloud->points[i].x;
            features[i].pt.y = in_cloud->points[i].y;
            features[i].pt.z = in_cloud->points[i].z;
            features[i].ptId = i;
            features[i].pt_num = search_indices.size();
            // debug  :
            if (in_cloud->points[i].x > max_x)
            {
                max_x = in_cloud->points[i].x;
            }
            // deprecated
            // 距离被查询点是否够近
            features[i].close_to_query_point.resize(search_indices.size());
            for (int j = 0; j < search_indices.size(); j++)
            {
                if (squared_distances[j] < 0.64 * radius * radius) // 0.5^(2/3)
                    features[i].close_to_query_point[j] = true;
                else
                    features[i].close_to_query_point[j] = false;
            }
            // 0.001~0.002ms
            get_pca_feature(in_cloud, search_indices, features[i]);

            if (features[i].pt_num > min_k)
                assign_normal(in_cloud->points[i], features[i]);
            // 释放内存
            std::vector<int>().swap(search_indices);
            std::vector<int>().swap(search_indices_used);
            std::vector<float>().swap(squared_distances);
        }
        //}
        // std::cout << "t_get_pca: " << t_get_pca.toc() << std::endl;
        // std::cout << "max_x: " << max_x << std::endl;

        return true;
    }

    /**
     * @brief
     *
     * @param in_cloud is the input Point Cloud Pointer
     * @param search_indices the neighborhood points' indices of the search point.
     * @param feature feature is the pca_feature_t of the search point.
     */
    bool get_pca_feature(pcl::PointCloud<PointINNO>::Ptr in_cloud, std::vector<int> &search_indices,
                         pca_feature_t &feature)
    {
        int pt_num = search_indices.size();

        if (pt_num <= 3)
            return false;

        pcl::PointCloud<PointINNO>::Ptr selected_cloud(new pcl::PointCloud<PointINNO>());
        for (int i = 0; i < pt_num; ++i)
            selected_cloud->points.push_back(in_cloud->points[search_indices[i]]);

        pcl::PCA<PointINNO> pca_operator;
        pca_operator.setInputCloud(selected_cloud);

        // Compute eigen values and eigen vectors
        Eigen::Matrix3f eigen_vectors = pca_operator.getEigenVectors();
        Eigen::Vector3f eigen_values = pca_operator.getEigenValues();

        feature.vectors.principalDirection = eigen_vectors.col(0);
        feature.vectors.normalDirection = eigen_vectors.col(2);

        feature.vectors.principalDirection.normalize();
        feature.vectors.normalDirection.normalize();

        feature.values.lamada1 = eigen_values(0);
        feature.values.lamada2 = eigen_values(1);
        feature.values.lamada3 = eigen_values(2);

        if ((feature.values.lamada1 + feature.values.lamada2 + feature.values.lamada3) == 0)
            feature.curvature = 0;
        else
            feature.curvature =
                feature.values.lamada3 / (feature.values.lamada1 + feature.values.lamada2 + feature.values.lamada3);

        // feature.linear_2 = (sqrt(feature.values.lamada1) -
        // sqrt(feature.values.lamada2)) / sqrt(feature.values.lamada1);
        // feature.planar_2 = (sqrt(feature.values.lamada2) -
        // sqrt(feature.values.lamada3)) / sqrt(feature.values.lamada1);
        // feature.spherical_2 = sqrt(feature.values.lamada3) /
        // sqrt(feature.values.lamada1);
        feature.linear_2 = ((feature.values.lamada1) - (feature.values.lamada2)) / (feature.values.lamada1);
        feature.planar_2 = ((feature.values.lamada2) - (feature.values.lamada3)) / (feature.values.lamada1);
        feature.spherical_2 = (feature.values.lamada3) / (feature.values.lamada1);

        search_indices.swap(feature.neighbor_indices);
        return true;
    }

    // is_palne_feature (true: assign point normal as pca normal vector, false: assign point normal as pca primary
    // direction vector)
    bool assign_normal(PointINNO &pt, pca_feature_t &pca_feature, bool is_plane_feature = true)
    {
        if (is_plane_feature)
        {

            pt.normal_x = pca_feature.vectors.normalDirection.x();
            pt.normal_y = pca_feature.vectors.normalDirection.y();
            pt.normal_z = pca_feature.vectors.normalDirection.z();
            pt.normal[3] = pca_feature.planar_2; // planrity
        }
        else
        {
            pt.normal_x = pca_feature.vectors.principalDirection.x();
            pt.normal_y = pca_feature.vectors.principalDirection.y();
            pt.normal_z = pca_feature.vectors.principalDirection.z();
            pt.normal[3] = pca_feature.linear_2; // linarity
        }
        return true;
    }

protected:
private:
    /**
     * \brief Check the Normals (if they are finite)
     * \param normals is the input Point Cloud (XYZI)'s Normal Pointer
     */
    void check_normal(pcl::PointCloud<pcl::Normal>::Ptr &normals)
    {
        // It is advisable to check the normals before the call to compute()
        for (int i = 0; i < normals->points.size(); i++)
        {
            if (!pcl::isFinite<pcl::Normal>(normals->points[i]))
            {
                normals->points[i].normal_x = 0.577; // 1/ sqrt(3)
                normals->points[i].normal_y = 0.577;
                normals->points[i].normal_z = 0.577;
                // normals->points[i].curvature = 0.0;
            }
        }
    }
};

class CProceesing
{
public:
    bool ground_filter_pmf(const pcl::PointCloud<PointINNO>::Ptr &cloud, pcl::PointCloud<PointINNO>::Ptr &gcloud,
                           pcl::PointCloud<PointINNO>::Ptr &ngcloud, int max_window_size, float slope,
                           float initial_distance, float max_distance)
    {
        pcl::PointIndicesPtr ground_points(new pcl::PointIndices);
        pcl::ProgressiveMorphologicalFilter<PointINNO> pmf;
        pmf.setInputCloud(cloud);
        pmf.setMaxWindowSize(max_window_size);    // 20
        pmf.setSlope(slope);                      // 1.0f
        pmf.setInitialDistance(initial_distance); // 0.5f
        pmf.setMaxDistance(max_distance);         // 3.0f
        pmf.extract(ground_points->indices);

        // Create the filtering object
        pcl::ExtractIndices<PointINNO> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(ground_points);
        extract.filter(*gcloud);

        // std::cout << "Ground cloud after filtering (PMF): " << std::endl;
        // std::cout << *gcloud << std::endl;

        // Extract non-ground returns
        extract.setNegative(true);
        extract.filter(*ngcloud);

        // std::out << "Non-ground cloud after filtering (PMF): " << std::endl;
        // std::out << *ngcloud << std::endl;

        return 1;
    }

    bool plane_seg_ransac(const pcl::PointCloud<PointINNO>::Ptr &cloud, float threshold, int max_iter,
                          pcl::PointCloud<PointINNO>::Ptr &planecloud,
                          pcl::ModelCoefficients::Ptr &coefficients) // Ransac
    {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        // Create the segmentation object
        pcl::SACSegmentation<PointINNO> sacseg;

        // Optional
        sacseg.setOptimizeCoefficients(true);

        // Mandatory
        sacseg.setModelType(pcl::SACMODEL_PLANE);
        sacseg.setMethodType(pcl::SAC_RANSAC);
        sacseg.setDistanceThreshold(threshold);
        sacseg.setMaxIterations(max_iter);

        sacseg.setInputCloud(cloud);
        sacseg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0)
        {
            PCL_ERROR("Could not estimate a planar model for the given dataset.");
        }

        /*cout << "Model coefficients: " << coefficients->values[0] << " "
        << coefficients->values[1] << " "
        << coefficients->values[2] << " "
        << coefficients->values[3] << std::endl;*/

        // LOG(INFO) << "Model inliers number: " << inliers->indices.size() << std::endl;

        for (size_t i = 0; i < inliers->indices.size(); ++i)
        {
            planecloud->push_back(cloud->points[inliers->indices[i]]);
        }
        return 1;
    }
};

bool estimate_ground_normal_by_ransac(pcl::PointCloud<PointINNO>::Ptr &grid_ground, float dist_thre, int max_iter,
                                      float &nx, float &ny, float &nz);

// fixed number random downsampling
// when keep_number == 0, the output point cloud would be empty (in other words, the input point cloud would be cleared)
bool random_downsample_pcl(pcl::PointCloud<PointINNO>::Ptr &cloud_in_out, int keep_number);

// fixed number random downsampling
// when keep_number == 0, the output point cloud would be empty
bool random_downsample_pcl(pcl::PointCloud<PointINNO>::Ptr &cloud_in, pcl::PointCloud<PointINNO>::Ptr &cloud_out,
                           int keep_number);

// extract stable points and then encode point cloud neighborhood feature descriptor (ncc: neighborhood category
// context) at the same time
bool encode_stable_points(const pcl::PointCloud<PointINNO>::Ptr &cloud_in, pcl::PointCloud<PointINNO>::Ptr &cloud_out,
                          const std::vector<pca_feature_t> &features, const std::vector<int> &index_with_feature,
                          float min_curvature, int min_feature_point_num_neighborhood, int min_point_num_neighborhood);

// Brief: Use NMS to select those key points having locally maximal curvature
bool non_max_suppress(pcl::PointCloud<PointINNO>::Ptr &cloud_in, pcl::PointCloud<PointINNO>::Ptr &cloud_out,
                      float nms_radius); // according to curvature

class FeatureExtract
{
public:
    PointCloudXYZI getFeature(pcl::PointCloud<PointINNO>::Ptr &in_pc);
    bool fast_ground_filter(
        const pcl::PointCloud<PointINNO>::Ptr &cloud_in, pcl::PointCloud<PointINNO>::Ptr &cloud_ground,
        pcl::PointCloud<PointINNO>::Ptr &cloud_ground_down, pcl::PointCloud<PointINNO>::Ptr &cloud_unground,
        pcl::PointCloud<PointINNO>::Ptr &cloud_curb, int min_grid_pt_num, float grid_resolution,
        float max_height_difference, float neighbor_height_diff, float max_ground_height, int ground_random_down_rate,
        int ground_random_down_down_rate, int nonground_random_down_rate, int reliable_neighbor_grid_num_thre,
        int estimate_ground_normal_method, float normal_estimation_radius, int distance_weight_downsampling_method,
        float standard_distance, bool fixed_num_downsampling, int down_ground_fixed_num, bool detect_curb_or_not,
        float intensity_thre, bool apply_grid_wise_outlier_filter);

    bool classify_nground_pts(
        pcl::PointCloud<PointINNO>::Ptr &cloud_in, pcl::PointCloud<PointINNO>::Ptr &cloud_pillar,
        pcl::PointCloud<PointINNO>::Ptr &cloud_beam, pcl::PointCloud<PointINNO>::Ptr &cloud_facade,
        pcl::PointCloud<PointINNO>::Ptr &cloud_roof, pcl::PointCloud<PointINNO>::Ptr &cloud_pillar_down,
        pcl::PointCloud<PointINNO>::Ptr &cloud_beam_down, pcl::PointCloud<PointINNO>::Ptr &cloud_facade_down,
        pcl::PointCloud<PointINNO>::Ptr &cloud_roof_down, pcl::PointCloud<PointINNO>::Ptr &cloud_vertex,
        float neighbor_searching_radius, int neighbor_k, int neigh_k_min,
        int pca_down_rate, // one in ${pca_down_rate} unground points would be
                           // select as the query points for calculating pca, the
                           // else would only be used as neighborhood points
        float edge_thre, float planar_thre, float edge_thre_down, float planar_thre_down, int extract_vertex_points_method,
        float curvature_thre, float vertex_curvature_non_max_radius, float linear_vertical_sin_high_thre,
        float linear_vertical_sin_low_thre, float planar_vertical_sin_high_thre, float planar_vertical_sin_low_thre,
        bool fixed_num_downsampling, int pillar_down_fixed_num, int facade_down_fixed_num,
        int beam_down_fixed_num, int roof_down_fixed_num, int unground_down_fixed_num,
        float beam_height_max, float roof_height_min, float feature_pts_ratio_guess,
        bool sharpen_with_nms, bool use_distance_adaptive_pca);
    void get_cloud_bbx(const pcl::PointCloud<PointINNO>::Ptr &cloud, bounds_t &bound);
    // Get Bound and Center of a Point Cloud
    void get_cloud_bbx_cpt(const pcl::PointCloud<PointINNO>::Ptr &cloud, bounds_t &bound, centerpoint_t &cp)
    {
        get_cloud_bbx(cloud, bound);
        cp.x = 0.5 * (bound.min_x + bound.max_x);
        cp.y = 0.5 * (bound.min_y + bound.max_y);
        cp.z = 0.5 * (bound.min_z + bound.max_z);
    }
    bool voxel_downsample(const pcl::PointCloud<PointINNO>::Ptr &cloud_in, pcl::PointCloud<PointINNO>::Ptr &cloud_out,
                          float voxel_size);
};

#endif // FEATURE_EXTRACT_H_