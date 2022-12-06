#include "feature_extract.h"

bool estimate_ground_normal_by_ransac(
    pcl::PointCloud<PointINNO>::Ptr &grid_ground, float dist_thre, int max_iter,
    float &nx, float &ny, float &nz) {
  CProceesing cpro;

  pcl::PointCloud<PointINNO>::Ptr grid_ground_fit(
      new pcl::PointCloud<PointINNO>);
  pcl::ModelCoefficients::Ptr grid_coeff(new pcl::ModelCoefficients);
  cpro.plane_seg_ransac(grid_ground, dist_thre, max_iter, grid_ground_fit,
                        grid_coeff);

  grid_ground.swap(grid_ground_fit);
  nx = grid_coeff->values[0];
  ny = grid_coeff->values[1];
  nz = grid_coeff->values[2];

  // LOG(INFO) << nx << "," << ny << "," << nz;
  return 1;
}

// fixed number random downsampling
// when keep_number == 0, the output point cloud would be empty (in other words,
// the input point cloud would be cleared)
bool random_downsample_pcl(pcl::PointCloud<PointINNO>::Ptr &cloud_in_out,
                           int keep_number) {
  if (cloud_in_out->points.size() <= keep_number)
    return false;
  else {
    if (keep_number == 0) {
      cloud_in_out.reset(new pcl::PointCloud<PointINNO>());
      return false;
    } else {
      pcl::PointCloud<PointINNO>::Ptr cloud_temp(
          new pcl::PointCloud<PointINNO>);
      pcl::RandomSample<PointINNO> ran_sample(true);  // Extract removed indices
      ran_sample.setInputCloud(cloud_in_out);
      ran_sample.setSample(keep_number);
      ran_sample.filter(*cloud_temp);
      cloud_temp->points.swap(cloud_in_out->points);
      return true;
    }
  }
}

// fixed number random downsampling
// when keep_number == 0, the output point cloud would be empty
bool random_downsample_pcl(pcl::PointCloud<PointINNO>::Ptr &cloud_in,
                           pcl::PointCloud<PointINNO>::Ptr &cloud_out,
                           int keep_number) {
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

  if (cloud_in->points.size() <= keep_number) {
    cloud_out = cloud_in;
    return false;
  } else {
    if (keep_number == 0)
      return false;
    else {
      pcl::RandomSample<PointINNO> ran_sample(true);  // Extract removed indices
      ran_sample.setInputCloud(cloud_in);
      ran_sample.setSample(keep_number);
      ran_sample.filter(*cloud_out);
      std::chrono::steady_clock::time_point toc =
          std::chrono::steady_clock::now();
      std::chrono::duration<double> time_used =
          std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
      // LOG(INFO) << "Random downsampling done in [" << time_used.count() *
      // 1000.0 << "] ms.";
      return true;
    }
  }
}

bool encode_stable_points(const pcl::PointCloud<PointINNO>::Ptr &cloud_in,
                          pcl::PointCloud<PointINNO>::Ptr &cloud_out,
                          const std::vector<pca_feature_t> &features,
                          const std::vector<int> &index_with_feature,
                          float min_curvature = 0.0,
                          int min_feature_point_num_neighborhood = 4,
                          int min_point_num_neighborhood = 8)
// extract stable points and then encode point cloud neighborhood feature
// descriptor (ncc: neighborhood category context) at the same time
{
  for (int i = 0; i < features.size(); ++i) {
    // float ratio1, ratio2;
    // ratio1 = features[i].values.lamada2 / features[i].values.lamada1;
    // ratio2 = features[i].values.lamada3 / features[i].values.lamada2;
    // if (ratio1 < stable_ratio_max && ratio2 < stable_ratio_max &&
    if (features[i].pt_num > min_point_num_neighborhood &&
        features[i].curvature > min_curvature) {
      float accu_intensity = 0.0;
      PointINNO pt;
      pt = cloud_in->points[i];
      pt.normal[3] = features[i].curvature;  // save in normal[3]

      int neighbor_total_count = 0, pillar_count = 0, beam_count = 0,
          facade_count = 0, roof_count = 0;
      int pillar_close_count = 0, pillar_far_count = 0, beam_close_count = 0,
          beam_far_count = 0, facade_close_count = 0, facade_far_count = 0,
          roof_close_count = 0, roof_far_count = 0;

      neighbor_total_count = features[i].neighbor_indices.size();

      for (int j = 0; j < neighbor_total_count; j++) {
        int temp_neighbor_index = features[i].neighbor_indices[j];
        switch (index_with_feature[temp_neighbor_index]) {
          case 1: {
            pillar_count++;
            if (features[i].close_to_query_point[j])
              pillar_close_count++;
            else
              pillar_far_count++;
            break;
          }
          case 2: {
            beam_count++;
            if (features[i].close_to_query_point[j])
              beam_close_count++;
            else
              beam_far_count++;
            break;
          }
          case 3: {
            facade_count++;
            if (features[i].close_to_query_point[j])
              facade_close_count++;
            else
              facade_far_count++;
            break;
          }
          case 4: {
            roof_count++;
            if (features[i].close_to_query_point[j])
              roof_close_count++;
            else
              roof_far_count++;
            break;
          }
          default:
            break;
        }
        accu_intensity += cloud_in->points[temp_neighbor_index].intensity;
      }
      if (pillar_count + beam_count + facade_count + roof_count <
          min_feature_point_num_neighborhood)
        continue;

      // TODO: it's a very stupid way to doing so, change the feature encoding
      // in code refactoring
      pillar_count = 100 * pillar_count / neighbor_total_count;
      beam_count = 100 * beam_count / neighbor_total_count;
      facade_count = 100 * facade_count / neighbor_total_count;
      roof_count = 100 * roof_count / neighbor_total_count;
      pillar_close_count = 100 * pillar_close_count / neighbor_total_count;
      beam_close_count = 100 * beam_close_count / neighbor_total_count;
      facade_close_count = 100 * facade_close_count / neighbor_total_count;
      roof_close_count = 100 * roof_close_count / neighbor_total_count;
      pillar_far_count = 100 * pillar_far_count / neighbor_total_count;
      beam_far_count = 100 * beam_far_count / neighbor_total_count;
      facade_far_count = 100 * facade_far_count / neighbor_total_count;
      roof_far_count = 100 * roof_far_count / neighbor_total_count;

      int descriptor = pillar_count * 1000000 + beam_count * 10000 +
                       facade_count * 100 +
                       roof_count;  // the neighborhood discriptor (8 numbers)
      int descriptor_1 = pillar_close_count * 1000000 +
                         beam_close_count * 10000 + facade_close_count * 100 +
                         roof_close_count;
      int descriptor_2 = pillar_far_count * 1000000 + beam_far_count * 10000 +
                         facade_far_count * 100 + roof_far_count;

      // TODO: fix later, keypoints would not be used in fine registration, so
      // we do not need the timestamp (stored in curvature) and normal vector
      pt.curvature = descriptor;
      pt.normal[0] = descriptor_1;
      pt.normal[1] = descriptor_2;

      pt.intensity =
          accu_intensity /
          neighbor_total_count;  // mean intensity of the nrighborhood
                                 // pt.normal[3] store the point curvature
                                 // pt.data[3] store the height of the point
                                 // above the ground

      //!!! TODO: fix, use customed point type, you need a lot of porperties for
      //! saving linearity, planarity,
      //! curvature, semantic label and timestamp
      //!!! However, within the template class, there might be a lot of problems
      //!(waiting for the code
      //! reproducing)

      cloud_out->points.push_back(pt);
    }
  }
  return true;
}

// Brief: Use NMS to select those key points having locally maximal curvature
bool non_max_suppress(pcl::PointCloud<PointINNO>::Ptr &cloud_in,
                      pcl::PointCloud<PointINNO>::Ptr &cloud_out,
                      float nms_radius)  // according to curvature
{
  bool distance_adaptive_on = false;
  float unit_dist = 35.0;
  bool kd_tree_already_built = false;
  const pcl::search::KdTree<PointINNO>::Ptr &built_tree = NULL;

  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  typename pcl::PointCloud<PointINNO>::Ptr cloud_temp(
      new pcl::PointCloud<PointINNO>());

  int pt_count_before = cloud_in->points.size();
  if (pt_count_before < 10) return false;

  std::sort(cloud_in->points.begin(), cloud_in->points.end(),
            [](const PointINNO &a, const PointINNO &b) {
              return a.normal[3] > b.normal[3];
            });  // using the unused normal[3] to represent what we want

  std::set<int, std::less<int>> unVisitedPtId;
  std::set<int, std::less<int>>::iterator iterUnseg;
  for (int i = 0; i < pt_count_before; ++i) unVisitedPtId.insert(i);

  typename pcl::search::KdTree<PointINNO>::Ptr tree(
      new pcl::search::KdTree<PointINNO>());
  if (kd_tree_already_built)
    tree = built_tree;
  else
    tree->setInputCloud(cloud_in);

  std::vector<int> search_indices;
  std::vector<float> distances;
  int keypointnum = 0;
  do {
    keypointnum++;
    std::vector<int>().swap(search_indices);
    std::vector<float>().swap(distances);

    int id;
    iterUnseg = unVisitedPtId.begin();
    id = *iterUnseg;
    cloud_out->points.push_back(cloud_in->points[id]);
    unVisitedPtId.erase(id);

    float non_max_radius = nms_radius;

    if (distance_adaptive_on) {
      double dist = std::sqrt(cloud_in->points[id].x * cloud_in->points[id].x +
                              cloud_in->points[id].y * cloud_in->points[id].y +
                              cloud_in->points[id].z * cloud_in->points[id].z);
      if (dist > unit_dist) {
        non_max_radius = std::sqrt(dist / unit_dist) * nms_radius;
        // neighborhood_k = (int)(unit_dist / dist * nearest_k));
      }
    }

    tree->radiusSearch(cloud_in->points[id], non_max_radius, search_indices,
                       distances);

    for (int i = 0; i < search_indices.size(); i++)
      unVisitedPtId.erase(search_indices[i]);

  } while (!unVisitedPtId.empty());

  int pt_count_after_nms = cloud_out->points.size();

  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

  // LOG(INFO) << "NMS done from [" << pt_count_before << "] to [" <<
  // pt_count_after_nms << "] points in [" << 1000.0
  // * time_used.count() << "] ms.";

  return true;
}

bool FeatureExtract::fast_ground_filter(
    const pcl::PointCloud<PointINNO>::Ptr &cloud_in,
    pcl::PointCloud<PointINNO>::Ptr &cloud_ground,
    pcl::PointCloud<PointINNO>::Ptr &cloud_ground_down,
    pcl::PointCloud<PointINNO>::Ptr &cloud_unground,
    pcl::PointCloud<PointINNO>::Ptr &cloud_curb, int min_grid_pt_num,
    float grid_resolution, float max_height_difference,
    float neighbor_height_diff, float max_ground_height,
    int ground_random_down_rate, int ground_random_down_down_rate,
    int nonground_random_down_rate, int reliable_neighbor_grid_num_thre,
    int estimate_ground_normal_method,
    float
        normal_estimation_radius,  // estimate_ground_normal_method, 0: directly
                                   // use (0,0,1), 1: estimate normal in fix
                                   // radius neighborhood , 2: estimate normal
                                   // in k nearest neighborhood, 3: use ransac
                                   // to estimate plane coeffs in a grid
    int distance_weight_downsampling_method,
    float standard_distance,  // standard distance: the distance where the
                              // distance_weight is 1
    bool fixed_num_downsampling = false, int down_ground_fixed_num = 1000,
    bool detect_curb_or_not = false, float intensity_thre = FLT_MAX,
    bool apply_grid_wise_outlier_filter =
        false)  // current intensity_thre is for kitti dataset (TODO: disable
                // it)
{
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

  PrincipleComponentAnalysis pca_estimator;

  pcl::PointCloud<PointINNO>::Ptr cloud_ground_full(
      new pcl::PointCloud<PointINNO>());

  int reliable_grid_pts_count_thre = min_grid_pt_num - 1;
  int count_checkpoint = 0;
  float sum_height = 0.001;
  float appro_mean_height;
  float min_ground_height = max_ground_height;
  float underground_noise_thre = -FLT_MAX;
  float non_ground_height_thre;
  float distance_weight;
  // int ground_random_down_rate_temp = ground_random_down_rate;
  // int nonground_random_down_rate_temp = nonground_random_down_rate;

  // For some points,  calculating the approximate mean height
  for (int j = 0; j < cloud_in->points.size(); j++) {
    if (j % 100 == 0) {
      sum_height += cloud_in->points[j].z;
      count_checkpoint++;
    }
  }
  appro_mean_height = sum_height / count_checkpoint;

  non_ground_height_thre = appro_mean_height + max_ground_height;
  // sometimes, there would be some underground ghost points (noise), however,
  // these points would be removed by scanner filter float
  // underground_noise_thre = appro_mean_height - max_ground_height;  // this
  // is a keyparameter.

  bounds_t bounds;
  centerpoint_t center_pt;
  this->get_cloud_bbx_cpt(
      cloud_in, bounds,
      center_pt);  // Inherited from its parent class, use this->

  // Construct Grid
  int row, col, num_grid;
  row = ceil((bounds.max_y - bounds.min_y) / grid_resolution);
  col = ceil((bounds.max_x - bounds.min_x) / grid_resolution);
  num_grid = row * col;

  std::chrono::steady_clock::time_point toc_1_1 =
      std::chrono::steady_clock::now();

  grid_t *grid = new grid_t[num_grid];

  // Each grid
  for (int i = 0; i < num_grid; i++) {
    grid[i].min_z = FLT_MAX;
    grid[i].neighbor_min_z = FLT_MAX;
  }

  // Each point ---> determine the grid to which the point belongs
  for (int j = 0; j < cloud_in->points.size(); j++) {
    int temp_row, temp_col, temp_id;
    temp_col = floor((cloud_in->points[j].x - bounds.min_x) / grid_resolution);
    temp_row = floor((cloud_in->points[j].y - bounds.min_y) / grid_resolution);
    temp_id = temp_row * col + temp_col;
    if (temp_id >= 0 && temp_id < num_grid) {
      if (distance_weight_downsampling_method > 0 && !grid[temp_id].pts_count) {
        grid[temp_id].dist2station =
            std::sqrt(cloud_in->points[j].x * cloud_in->points[j].x +
                      cloud_in->points[j].y * cloud_in->points[j].y +
                      cloud_in->points[j].z * cloud_in->points[j].z);
      }

      if (cloud_in->points[j].z > non_ground_height_thre) {
        distance_weight = 1.0 * standard_distance /
                          (grid[temp_id].dist2station +
                           0.0001);  // avoiding Floating point exception
        int nonground_random_down_rate_temp = nonground_random_down_rate;
        if (distance_weight_downsampling_method == 1)  // linear weight
          nonground_random_down_rate_temp =
              (int)(distance_weight * nonground_random_down_rate + 1);
        else if (distance_weight_downsampling_method == 2)  // quadratic weight
          nonground_random_down_rate_temp =
              (int)(distance_weight * distance_weight *
                        nonground_random_down_rate +
                    1);

        if (j % nonground_random_down_rate_temp == 0 ||
            cloud_in->points[j].intensity > intensity_thre) {
          cloud_in->points[j].data[3] =
              cloud_in->points[j].z -
              (appro_mean_height - 3.0);  // data[3] stores the approximate
                                          // point height above ground
          cloud_unground->points.push_back(cloud_in->points[j]);
        }
      } else if (cloud_in->points[j].z > underground_noise_thre) {
        grid[temp_id].pts_count++;
        grid[temp_id].point_id.push_back(j);
        if (cloud_in->points[j].z < grid[temp_id].min_z)  //
        {
          grid[temp_id].min_z = cloud_in->points[j].z;
          grid[temp_id].neighbor_min_z = cloud_in->points[j].z;
        }
      }
    }
  }
  std::chrono::steady_clock::time_point toc_1_2 =
      std::chrono::steady_clock::now();

  // if (apply_grid_wise_outlier_filter)
  // {
  //     // Each grid: Check outlier //calculate mean and standard deviation of
  //     z
  //     // in one grid, then set mean-2*std as the threshold for outliers
  //     for (int i = 0; i < num_grid; i++)
  //     {
  //         if (grid[i].pts_count >= min_grid_pt_num)
  //         {
  //             double sum_z = 0, sum_z2 = 0, std_z = 0, mean_z = 0;
  //             for (int j = 0; j < grid[i].point_id.size(); j++)
  //                 sum_z += cloud_in->points[grid[i].point_id[j]].z;
  //             mean_z = sum_z / grid[i].pts_count;
  //             for (int j = 0; j < grid[i].point_id.size(); j++)
  //                 sum_z2 += (cloud_in->points[grid[i].point_id[j]].z -
  //                 mean_z) *
  //                           (cloud_in->points[grid[i].point_id[j]].z -
  //                           mean_z);
  //             std_z = std::sqrt(sum_z2 / grid[i].pts_count);
  //             grid[i].min_z_outlier_thre = mean_z - outlier_std_scale *
  //             std_z; grid[i].min_z = max_(grid[i].min_z,
  //             grid[i].min_z_outlier_thre); grid[i].neighbor_min_z =
  //             grid[i].min_z;
  //         }
  //     }
  // }

  std::chrono::steady_clock::time_point toc_1_3 =
      std::chrono::steady_clock::now();

  // Each grid
  for (int m = 0; m < num_grid; m++) {
    int temp_row, temp_col;
    temp_row = m / col;
    temp_col = m % col;
    if (temp_row >= 1 && temp_row <= row - 2 && temp_col >= 1 &&
        temp_col <= col - 2) {
      for (int j = -1; j <= 1; j++)  // row
      {
        for (int k = -1; k <= 1; k++)  // col
        {
          grid[m].neighbor_min_z =
              min_(grid[m].neighbor_min_z, grid[m + j * col + k].min_z);
          if (grid[m + j * col + k].pts_count > reliable_grid_pts_count_thre)
            grid[m].reliable_neighbor_grid_num++;
        }
      }
    }
  }

  double consuming_time_ransac = 0.0;

  std::chrono::steady_clock::time_point toc_1_4 =
      std::chrono::steady_clock::now();

  std::vector<pcl::PointCloud<PointINNO>::Ptr> grid_ground_pcs(num_grid);
  std::vector<pcl::PointCloud<PointINNO>::Ptr> grid_unground_pcs(num_grid);
  for (int i = 0; i < num_grid; i++) {
    pcl::PointCloud<PointINNO>::Ptr grid_ground_pc_temp(
        new pcl::PointCloud<PointINNO>);
    grid_ground_pcs[i] = grid_ground_pc_temp;
    pcl::PointCloud<PointINNO>::Ptr grid_unground_pc_temp(
        new pcl::PointCloud<PointINNO>);
    grid_unground_pcs[i] = grid_unground_pc_temp;
  }

  std::chrono::steady_clock::time_point toc_1 =
      std::chrono::steady_clock::now();

  // For each grid
  //     omp_set_num_threads(min_(6, omp_get_max_threads()));
  // #pragma omp parallel for
  for (int i = 0; i < num_grid; i++) {
    pcl::PointCloud<PointINNO>::Ptr grid_ground(new pcl::PointCloud<PointINNO>);
    // Filtering some grids with too little points
    if (grid[i].pts_count >= min_grid_pt_num &&
        grid[i].reliable_neighbor_grid_num >= reliable_neighbor_grid_num_thre) {
      int ground_random_down_rate_temp = ground_random_down_rate;
      int nonground_random_down_rate_temp = nonground_random_down_rate;
      distance_weight =
          1.0 * standard_distance / (grid[i].dist2station + 0.0001);
      if (distance_weight_downsampling_method == 1)  // linear weight
      {
        ground_random_down_rate_temp =
            (int)(distance_weight * ground_random_down_rate + 1);
        nonground_random_down_rate_temp =
            (int)(distance_weight * nonground_random_down_rate + 1);
      } else if (distance_weight_downsampling_method == 2)  // quadratic weight
      {
        ground_random_down_rate_temp =
            (int)(distance_weight * distance_weight * ground_random_down_rate +
                  1);
        nonground_random_down_rate_temp =
            (int)(distance_weight * distance_weight *
                      nonground_random_down_rate +
                  1);
      }
      // LOG(WARNING) << ground_random_down_rate_temp << "," <<
      // nonground_random_down_rate_temp;
      if (grid[i].min_z - grid[i].neighbor_min_z < neighbor_height_diff) {
        for (int j = 0; j < grid[i].point_id.size(); j++) {
          if (cloud_in->points[grid[i].point_id[j]].z >
              grid[i].min_z_outlier_thre) {
            if (cloud_in->points[grid[i].point_id[j]].z - grid[i].min_z <
                max_height_difference) {
              // cloud_ground_full->points.push_back(cloud_in->points[grid[i].point_id[j]]);
              if (estimate_ground_normal_method == 3)
                grid_ground->points.push_back(
                    cloud_in->points[grid[i].point_id[j]]);
              else {
                if (j % ground_random_down_rate_temp == 0)  // for example 10
                {
                  if (estimate_ground_normal_method == 0) {
                    cloud_in->points[grid[i].point_id[j]].normal_x = 0.0;
                    cloud_in->points[grid[i].point_id[j]].normal_y = 0.0;
                    cloud_in->points[grid[i].point_id[j]].normal_z = 1.0;
                  }
                  grid_ground_pcs[i]->points.push_back(
                      cloud_in->points[grid[i].point_id[j]]);
                  // cloud_ground->points.push_back(cloud_in->points[grid[i].point_id[j]]);
                  // //Add to ground points
                }
              }
            } else  // inner grid unground points
            {
              if (j % nonground_random_down_rate_temp == 0 ||
                  cloud_in->points[grid[i].point_id[j]].intensity >
                      intensity_thre)  // extract more points
                                       // on signs and vehicle
                                       // license plate
              {
                cloud_in->points[grid[i].point_id[j]].data[3] =
                    cloud_in->points[grid[i].point_id[j]].z -
                    grid[i].min_z;  // data[3] stores the point
                                    // height above ground
                grid_unground_pcs[i]->points.push_back(
                    cloud_in->points[grid[i].point_id[j]]);
                // cloud_unground->points.push_back(cloud_in->points[grid[i].point_id[j]]);
                // //Add to nonground points
              }
            }
          }
        }
      } else  // unground grid
      {
        for (int j = 0; j < grid[i].point_id.size(); j++) {
          if (cloud_in->points[grid[i].point_id[j]].z >
                  grid[i].min_z_outlier_thre &&
              (j % nonground_random_down_rate_temp == 0 ||
               cloud_in->points[grid[i].point_id[j]].intensity >
                   intensity_thre)) {
            cloud_in->points[grid[i].point_id[j]].data[3] =
                cloud_in->points[grid[i].point_id[j]].z -
                grid[i].neighbor_min_z;  // data[3] stores the point
                                         // height above ground
            grid_unground_pcs[i]->points.push_back(
                cloud_in->points[grid[i].point_id[j]]);
            // cloud_unground->points.push_back(cloud_in->points[grid[i].point_id[j]]);
            // //Add to nonground points
          }
        }
      }
      if (estimate_ground_normal_method == 3 &&
          grid_ground->points.size() >= min_grid_pt_num) {
        std::chrono::steady_clock::time_point tic_ransac =
            std::chrono::steady_clock::now();
        float normal_x, normal_y, normal_z;

        // RANSAC iteration number equation: p=1-(1-r^N)^M,
        // r is the inlier ratio (> 0.75 in our case), N is 3 in our case
        // (3 points can fit a plane), to get a confidence > 0.99, we
        // need about 20 iteration (M=20)
        estimate_ground_normal_by_ransac(grid_ground,
                                         0.3 * max_height_difference, 20,
                                         normal_x, normal_y, normal_z);

        for (int j = 0; j < grid_ground->points.size(); j++) {
          if (j % ground_random_down_rate_temp == 0 &&
              std::abs(normal_z) > 0.8)  // 53 deg
          {
            grid_ground->points[j].normal_x = normal_x;
            grid_ground->points[j].normal_y = normal_y;
            grid_ground->points[j].normal_z = normal_z;
            grid_ground_pcs[i]->points.push_back(
                grid_ground->points
                    [j]);  // Add to ground points
                           // cloud_ground->points.push_back(grid_ground->points[j]);
                           // //Add to ground points
          }
        }
        std::chrono::steady_clock::time_point toc_ransac =
            std::chrono::steady_clock::now();
        std::chrono::duration<double> ground_ransac_time_per_grid =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                toc_ransac - tic_ransac);
        consuming_time_ransac +=
            ground_ransac_time_per_grid.count() * 1000.0;  // unit: ms
      }
      pcl::PointCloud<PointINNO>().swap(*grid_ground);
    }
  }

  // combine the ground and unground points
  for (int i = 0; i < num_grid; i++) {
    cloud_ground->points.insert(cloud_ground->points.end(),
                                grid_ground_pcs[i]->points.begin(),
                                grid_ground_pcs[i]->points.end());
    cloud_unground->points.insert(cloud_unground->points.end(),
                                  grid_unground_pcs[i]->points.begin(),
                                  grid_unground_pcs[i]->points.end());
  }

  // free memory
  delete[] grid;

  std::chrono::steady_clock::time_point toc_2 =
      std::chrono::steady_clock::now();

  int normal_estimation_neighbor_k = 2 * min_grid_pt_num;
  pcl::PointCloud<pcl::Normal>::Ptr ground_normal(
      new pcl::PointCloud<pcl::Normal>);
  if (estimate_ground_normal_method == 1)
    pca_estimator.get_normal_pcar(cloud_ground, normal_estimation_radius,
                                  ground_normal);
  else if (estimate_ground_normal_method == 2)
    pca_estimator.get_normal_pcak(cloud_ground, normal_estimation_neighbor_k,
                                  ground_normal);

  for (int i = 0; i < cloud_ground->points.size(); i++) {
    if (estimate_ground_normal_method == 1 ||
        estimate_ground_normal_method == 2) {
      cloud_ground->points[i].normal_x = ground_normal->points[i].normal_x;
      cloud_ground->points[i].normal_y = ground_normal->points[i].normal_y;
      cloud_ground->points[i].normal_z = ground_normal->points[i].normal_z;
    }
    if (!fixed_num_downsampling) {
      // std::cout << "\n"<<cloud_ground->points[i].normal_x << "," <<
      // cloud_ground->points[i].normal_y << "," <<
      // cloud_ground->points[i].normal_z << std::endl;
      if (i % ground_random_down_down_rate == 0)
        cloud_ground_down->points.push_back(cloud_ground->points[i]);
    }
  }

  if (fixed_num_downsampling)
    random_downsample_pcl(cloud_ground, cloud_ground_down,
                          down_ground_fixed_num);

  pcl::PointCloud<pcl::Normal>().swap(*ground_normal);

  std::chrono::steady_clock::time_point toc_3 =
      std::chrono::steady_clock::now();
  std::chrono::duration<double> ground_seg_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc_2 - tic);
  std::chrono::duration<double> ground_seg_prepare_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc_1 - tic);
  std::chrono::duration<double> ground_normal_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc_3 - toc_2);

  // std::chrono::duration<double> prepare_1 =
  // std::chrono::duration_cast<std::chrono::duration<double>>(toc_1_1 - tic);
  // std::chrono::duration<double> prepare_2 =
  // std::chrono::duration_cast<std::chrono::duration<double>>(toc_1_2 -
  // toc_1_1); std::chrono::duration<double> prepare_3 =
  // std::chrono::duration_cast<std::chrono::duration<double>>(toc_1_3 -
  // toc_1_2); std::chrono::duration<double> prepare_4 =
  // std::chrono::duration_cast<std::chrono::duration<double>>(toc_1_4 -
  // toc_1_3); std::chrono::duration<double> prepare_5 =
  // std::chrono::duration_cast<std::chrono::duration<double>>(toc_1 -
  // toc_1_4);

  std::cout << "\n"
            << "Ground: [" << cloud_ground->points.size() << " | "
            << cloud_ground_down->points.size() << "] Unground: ["
            << cloud_unground->points.size() << "]." << std::endl;

  if (estimate_ground_normal_method == 3) {
    // std::cout << "\n" << "Ground segmentation done in [" <<
    // ground_seg_time.count() * 1000.0 - consuming_time_ransac << "] ms." <<
    // std::endl; std::cout << "\n" << "Ground Normal Estimation done in [" <<
    // consuming_time_ransac << "] ms." << std::endl;
    std::cout << "\n"
              << "Ground segmentation and normal estimation in ["
              << ground_seg_time.count() * 1000.0 << "] ms."
              << ",in which preparation costs ["
              << ground_seg_prepare_time.count() * 1000.0 << "] ms."
              << std::endl;
    // output detailed consuming time
    // std::cout << "\n" << prepare_1.count() * 1000.0 << "," <<
    // prepare_2.count() * 1000.0 << "," << prepare_3.count() * 1000.0 << "," <<
    // prepare_4.count() * 1000.0 << "," << prepare_5.count() * 1000.0;
  } else {
    std::cout << "\n"
              << "Ground segmentation done in ["
              << ground_seg_time.count() * 1000.0 << "] ms." << std::endl;
    std::cout << "\n"
              << "Ground Normal Estimation done in ["
              << ground_normal_time.count() * 1000.0 << "] ms."
              << " preparation in [" << ground_seg_prepare_time.count() * 1000.0
              << "] ms." << std::endl;
  }
#if 0  // curb detection (deprecated)
			if (detect_curb_or_not)
			{
				//detect curb points
				std::vector<pca_feature_t> curb_pca_features;
				 pcl::PointCloud<PointINNO>::Ptr cloud_curb_candidate(new pcl::PointCloud<PointINNO>());
				float pca_radius_curb = normal_estimation_radius;
				int pca_k_curb = normal_estimation_neighbor_k;
				int pca_min_pt_num = 4;
				float curb_linearity_thre = 0.7;
				float direction_z_max = 0.1;

				detect_curbs(cloud_ground_full, cloud_curb_candidate);

				pca_estimator.get_pc_pca_feature(cloud_curb_candidate, curb_pca_features, pca_radius_curb, pca_k_curb);
				for (int i = 0; i < cloud_curb_candidate->points.size(); i++)
				{
					if (curb_pca_features[i].pt_num >= pca_min_pt_num &&
						curb_pca_features[i].linear_2 > curb_linearity_thre &&
						std::abs(curb_pca_features[i].vectors.principalDirection.z()) < direction_z_max)
					{
						pca_estimator.assign_normal(cloud_curb_candidate->points[i], curb_pca_features[i], false); //assign primary direction vector
						cloud_curb->points.push_back(cloud_curb_candidate->points[i]);
					}
				}

				pcl::PointCloud<PointINNO>().swap(*cloud_curb_candidate);
				std::vector<pca_feature_t>().swap(curb_pca_features);

				std::chrono::steady_clock::time_point toc_3 = std::chrono::steady_clock::now();

				std::chrono::duration<double> curb_time = std::chrono::duration_cast<std::chrono::duration<double>>(toc_3 - toc_2);

				std::cout << "\n"<< "[" << cloud_curb->points.size() << "] curb points detected in [" << curb_time.count() * 1000.0 << "] ms." << std::endl;
			}
#endif
  // pcl::PointCloud<PointINNO>().swap(*cloud_ground_full);
  return 1;
}

// Brief: Classfiy the downsampled non-ground points into several types
// (Pillar, Beam, Facade, Roof, Vertex) according to the pca features
// (combination of eigen values and eigen vectors)
bool FeatureExtract::classify_nground_pts(
    pcl::PointCloud<PointINNO>::Ptr &cloud_in,
    pcl::PointCloud<PointINNO>::Ptr &cloud_pillar,
    pcl::PointCloud<PointINNO>::Ptr &cloud_beam,
    pcl::PointCloud<PointINNO>::Ptr &cloud_facade,
    pcl::PointCloud<PointINNO>::Ptr &cloud_roof,
    pcl::PointCloud<PointINNO>::Ptr &cloud_pillar_down,
    pcl::PointCloud<PointINNO>::Ptr &cloud_beam_down,
    pcl::PointCloud<PointINNO>::Ptr &cloud_facade_down,
    pcl::PointCloud<PointINNO>::Ptr &cloud_roof_down,
    pcl::PointCloud<PointINNO>::Ptr &cloud_vertex,
    float neighbor_searching_radius, int neighbor_k, int neigh_k_min,
    int pca_down_rate,  // one in ${pca_down_rate} unground points would be
                        // select as the query points for calculating pca, the
                        // else would only be used as neighborhood points
    float edge_thre, float planar_thre, float edge_thre_down,
    float planar_thre_down, int extract_vertex_points_method,
    float curvature_thre, float vertex_curvature_non_max_radius,
    float linear_vertical_sin_high_thre, float linear_vertical_sin_low_thre,
    float planar_vertical_sin_high_thre, float planar_vertical_sin_low_thre,
    bool fixed_num_downsampling = true, int pillar_down_fixed_num = 200,
    int facade_down_fixed_num = 800, int beam_down_fixed_num = 200,
    int roof_down_fixed_num = 100, int unground_down_fixed_num = 20000,
    float beam_height_max = FLT_MAX, float roof_height_min = -FLT_MAX,
    float feature_pts_ratio_guess = 0.3, bool sharpen_with_nms = false,
    bool use_distance_adaptive_pca = false) {
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  //   TicToc t_downsamp;
  std::cout << "size before down: " << cloud_in->size() << std::endl;
  if (fixed_num_downsampling)
    random_downsample_pcl(cloud_in, unground_down_fixed_num);
  //   std::cout << "t_downsamp : " << t_downsamp.toc()
  // << "  size after down: " << cloud_in->size() << std::endl;
  std::cout << "  size after down: " << cloud_in->size() << std::endl;
  // Do PCA
  PrincipleComponentAnalysis pca_estimator;
  std::vector<pca_feature_t> cloud_features;

  pcl::KdTreeFLANN<PointINNO>::Ptr tree(new pcl::KdTreeFLANN<PointINNO>);
  tree->setInputCloud(cloud_in);

  float unit_distance = 30.0;
  pca_estimator.get_pc_pca_feature(
      cloud_in, cloud_features, tree, neighbor_searching_radius, neighbor_k, 1,
      pca_down_rate, use_distance_adaptive_pca, unit_distance);
  // LOG(WARNING)<< "PCA done";

  std::chrono::steady_clock::time_point toc_pca =
      std::chrono::steady_clock::now();

  // the radius should be larger for far away points
  std::vector<int> index_with_feature(
      cloud_in->points.size(),
      0);  // 0 - not special points, 1 - pillar, 2 - beam, 3 - facade, 4 - roof

  for (int i = 0; i < cloud_in->points.size(); i++) {
    if (cloud_features[i].pt_num > neigh_k_min) {
      // 边缘特征：pillar / beam
      if (cloud_features[i].linear_2 > edge_thre) {
        if (std::abs(cloud_features[i].vectors.principalDirection.z()) >
            linear_vertical_sin_high_thre) {
          pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i],
                                      false);
          cloud_pillar->points.push_back(cloud_in->points[i]);
          index_with_feature[i] = 1;
        } else if (std::abs(cloud_features[i].vectors.principalDirection.z()) <
                       linear_vertical_sin_low_thre &&
                   cloud_in->points[i].z < beam_height_max) {
          pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i],
                                      false);
          cloud_beam->points.push_back(cloud_in->points[i]);
          index_with_feature[i] = 2;
        } else {
          ;
        }
        // 降采样pillar / beam
        // if (!sharpen_with_nms && cloud_features[i].linear_2 > edge_thre_down)
        // {
        //     if (std::abs(cloud_features[i].vectors.principalDirection.z()) >
        //     linear_vertical_sin_high_thre)
        //         cloud_pillar_down->points.push_back(cloud_in->points[i]);
        //     else if
        //     (std::abs(cloud_features[i].vectors.principalDirection.z()) <
        //                  linear_vertical_sin_low_thre &&
        //              cloud_in->points[i].z < beam_height_max)
        //         cloud_beam_down->points.push_back(cloud_in->points[i]);
        //     else
        //     {
        //         ;
        //     }
        // }
      }
      // 平面类型：facade / roof
      else if (cloud_features[i].planar_2 > planar_thre) {
        if (std::abs(cloud_features[i].vectors.normalDirection.z()) >
                planar_vertical_sin_high_thre &&
            cloud_in->points[i].z > roof_height_min) {
          pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i],
                                      true);
          cloud_roof->points.push_back(cloud_in->points[i]);
          index_with_feature[i] = 4;
        } else if (std::abs(cloud_features[i].vectors.normalDirection.z()) <
                   planar_vertical_sin_low_thre) {
          pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i],
                                      true);
          cloud_facade->points.push_back(cloud_in->points[i]);
          index_with_feature[i] = 3;
        } else {
          ;
        }
        //降采样
        // if (!sharpen_with_nms && cloud_features[i].planar_2 >
        // planar_thre_down)
        // {
        //     if (std::abs(cloud_features[i].vectors.normalDirection.z()) >
        //     planar_vertical_sin_high_thre &&
        //         cloud_in->points[i].z > roof_height_min)
        //         cloud_roof_down->points.push_back(cloud_in->points[i]);
        //     else if (std::abs(cloud_features[i].vectors.normalDirection.z())
        //     < planar_vertical_sin_low_thre)
        //         cloud_facade_down->points.push_back(cloud_in->points[i]);
        //     else
        //     {
        //         ;
        //     }
        // }
      }
    }
  }

  // According to the parameter 'extract_vertex_points_method' (0,1,2...)
  // set stablilty_thre as 0 to disable the vertex extraction
  if (curvature_thre < 1e-8) extract_vertex_points_method = 0;

  // Find Edge points by picking high curvature points among the neighborhood
  // of unground geometric feature points (2)
  if (extract_vertex_points_method == 2) {
    float vertex_feature_ratio_thre = feature_pts_ratio_guess / pca_down_rate;
    for (int i = 0; i < cloud_in->points.size(); i++) {
      // if (index_with_feature[i] == 0)
      // 	cloud_vertex->points.push_back(cloud_in->points[i]);

      if (index_with_feature[i] == 0 &&
          cloud_features[i].pt_num > neigh_k_min &&
          cloud_features[i].curvature > curvature_thre)  // curvature_thre means
                                                         // curvature_thre here
      {
        int geo_feature_point_count = 0;
        for (int j = 0; j < cloud_features[i].neighbor_indices.size(); j++) {
          if (index_with_feature[cloud_features[i].neighbor_indices[j]])
            geo_feature_point_count++;
        }
        // std::cout<< "facade neighbor num: " <<geo_feature_point_count <<
        // std::endl;
        if (1.0 * geo_feature_point_count / cloud_features[i].pt_num >
            vertex_feature_ratio_thre)  // most of the neighbors are
                                        // feature points
        {
          // cloud_vertex->points.push_back(cloud_in->points[i]);

          pca_estimator.assign_normal(cloud_in->points[i], cloud_features[i],
                                      false);
          cloud_in->points[i].normal[3] =
              5.0 * cloud_features[i].curvature;  // save in the un-used
                                                  // normal[3] (PointNormal4D)
          if (std::abs(cloud_features[i].vectors.principalDirection.z()) >
              linear_vertical_sin_high_thre) {
            cloud_pillar->points.push_back(cloud_in->points[i]);
            // cloud_pillar_down->points.push_back(cloud_in->points[i]);
            index_with_feature[i] = 1;
          } else if (std::abs(
                         cloud_features[i].vectors.principalDirection.z()) <
                         linear_vertical_sin_low_thre &&
                     cloud_in->points[i].z < beam_height_max) {
            cloud_beam->points.push_back(cloud_in->points[i]);
            // cloud_beam_down->points.push_back(cloud_in->points[i]);
            index_with_feature[i] = 2;
          }
        }
      }
    }
  }

  // if extract_vertex_points_method == 0 ---> do not extract vertex points (0)
  std::chrono::steady_clock::time_point toc_1 =
      std::chrono::steady_clock::now();
  // extract neighborhood feature descriptor for pillar points
  // Find Vertex (Edge) points by picking points with maximum local curvature
  // (1) if (extract_vertex_points_method == 1) //Deprecated
  // detect_key_pts(cloud_in, cloud_features,
  // index_with_feature,cloud_vertex, 4.0 * curvature_thre,
  // vertex_curvature_non_max_radius, 0.5 * curvature_thre);
  int min_neighbor_feature_pts =
      (int)(feature_pts_ratio_guess / pca_down_rate * neighbor_k) - 1;

  // get the vertex keypoints and encode its neighborhood in a simple
  // descriptor
  encode_stable_points(
      cloud_in, cloud_vertex, cloud_features, index_with_feature,
      0.3 * curvature_thre, min_neighbor_feature_pts,
      neigh_k_min);  // encode the keypoints, we will get a simple descriptor of
                     // the putable keypoints

  // LOG(WARNING)<< "encode ncc feature descriptor done";

  std::chrono::steady_clock::time_point toc_2 =
      std::chrono::steady_clock::now();

  // Non_max_suppression of the feature points //TODO: add already built-kd
  // tree here
  if (sharpen_with_nms) {
    float nms_radius = 0.25 * neighbor_searching_radius;
#pragma omp parallel sections
    {
#pragma omp section
      {
        if (pillar_down_fixed_num > 0)
          non_max_suppress(cloud_pillar, cloud_pillar_down, nms_radius);
      }
#pragma omp section
      {
        if (facade_down_fixed_num > 0)
          non_max_suppress(cloud_facade, cloud_facade_down, nms_radius);
      }
#pragma omp section
      {
        if (beam_down_fixed_num > 0)
          non_max_suppress(cloud_beam, cloud_beam_down, nms_radius);

        if (roof_down_fixed_num > 0)
          non_max_suppress(cloud_roof, cloud_roof_down, nms_radius);
      }
    }
  }

  std::chrono::steady_clock::time_point toc_3 =
      std::chrono::steady_clock::now();

  // if (fixed_num_downsampling)
  // {
  //     random_downsample_pcl(cloud_pillar_down, pillar_down_fixed_num);
  //     int sector_num = 4;
  //     xy_normal_balanced_downsample(cloud_facade_down,
  //     (int)(facade_down_fixed_num / sector_num), sector_num);

  //     xy_normal_balanced_downsample(cloud_beam_down,
  //     (int)(beam_down_fixed_num / sector_num),
  //                                   sector_num); // here the normal is the
  //                                   primary vector
  //                                                //
  //                                                random_downsample_pcl(cloud_roof_down,
  //                                                100);
  //     random_downsample_pcl(cloud_roof_down, roof_down_fixed_num);
  // }

  std::chrono::steady_clock::time_point toc_4 =
      std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used_pca =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc_pca - tic);
  std::chrono::duration<double> time_used_extract_geo_pts =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc_1 -
                                                                toc_pca);
  std::chrono::duration<double> time_used_encoding_key_pts =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc_2 - toc_1);
  std::chrono::duration<double> time_used_nms_sharping =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc_3 - toc_2);
  std::chrono::duration<double> time_used_fixed_num_downsampling =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc_4 - toc_3);
  std::chrono::duration<double> time_used_all =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc_4 - tic);

  // Free the memory
  std::vector<pca_feature_t>().swap(cloud_features);
  std::vector<int>().swap(index_with_feature);

  std::cout << "\n"
            << "Unground geometric feature points extracted done in ["
            << time_used_all.count() * 1000.0 << "] ms." << std::endl;
  std::cout << "\n"
            << "Details: pca in [" << time_used_pca.count() * 1000.0
            << "] ms, geometric feature points extracted in ["
            << time_used_extract_geo_pts.count() * 1000.0
            << "] ms, encoding keypoints in ["
            << time_used_encoding_key_pts.count() * 1000.0
            << "] ms, nms sharpen in ["
            << time_used_nms_sharping.count() * 1000.0
            << "] ms, downsampling in ["
            << time_used_fixed_num_downsampling.count() * 1000.0 << "] ms."
            << std::endl;
  std::cout << "\n"
            << "Pillar: [" << cloud_pillar->points.size() << " | "
            << cloud_pillar_down->points.size() << "] Beam: ["
            << cloud_beam->points.size() << " | "
            << cloud_beam_down->points.size() << "] Facade: ["
            << cloud_facade->points.size() << " | "
            << cloud_facade_down->points.size() << "] Roof: ["
            << cloud_roof->points.size() << " | "
            << cloud_roof_down->points.size() << "] Vertex: ["
            << cloud_vertex->points.size() << "]." << std::endl;

  return 1;
}

void FeatureExtract::get_cloud_bbx(const pcl::PointCloud<PointINNO>::Ptr &cloud,
                                   bounds_t &bound) {
  double min_x = DBL_MAX;
  double min_y = DBL_MAX;
  double min_z = DBL_MAX;
  double max_x = -DBL_MAX;
  double max_y = -DBL_MAX;
  double max_z = -DBL_MAX;

  for (int i = 0; i < cloud->points.size(); i++) {
    if (min_x > cloud->points[i].x) min_x = cloud->points[i].x;
    if (min_y > cloud->points[i].y) min_y = cloud->points[i].y;
    if (min_z > cloud->points[i].z) min_z = cloud->points[i].z;
    if (max_x < cloud->points[i].x) max_x = cloud->points[i].x;
    if (max_y < cloud->points[i].y) max_y = cloud->points[i].y;
    if (max_z < cloud->points[i].z) max_z = cloud->points[i].z;
  }
  bound.min_x = min_x;
  bound.max_x = max_x;
  bound.min_y = min_y;
  bound.max_y = max_y;
  bound.min_z = min_z;
  bound.max_z = max_z;
}

bool FeatureExtract::voxel_downsample(
    const pcl::PointCloud<PointINNO>::Ptr &cloud_in,
    pcl::PointCloud<PointINNO>::Ptr &cloud_out, float voxel_size) {
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

  // You can set the downsampling_radius as 0 to disable the downsampling in
  // order to save time for lidar odometry test
  if (voxel_size < 0.001) {
    std::cout << "\n"
              << "too small voxel size, the downsampling would be disabled"
              << std::endl;
    cloud_out = cloud_in;

    // for (int i=0; i<cloud_out->points.size(); i++)
    //   std::cout << "\n"<< i <<","<< (int)(cloud_out->points[i].curvature);
    return false;
  }

  float inverse_voxel_size = 1.0f / voxel_size;

  Eigen::Vector4f min_p, max_p;
  pcl::getMinMax3D(*cloud_in, min_p, max_p);

  Eigen::Vector4f gap_p;  // boundingbox gap;
  gap_p = max_p - min_p;

  unsigned long long max_vx = ceil(gap_p.coeff(0) * inverse_voxel_size) + 1;
  unsigned long long max_vy = ceil(gap_p.coeff(1) * inverse_voxel_size) + 1;
  unsigned long long max_vz = ceil(gap_p.coeff(2) * inverse_voxel_size) + 1;

  if (max_vx * max_vy * max_vz >=
      std::numeric_limits<unsigned long long>::max()) {
    std::cout << "Filtering Failed: The number of box exceed the limit."
              << std::endl;
    return 0;
  }

  unsigned long long mul_vx = max_vy * max_vz;
  unsigned long long mul_vy = max_vz;
  unsigned long long mul_vz = 1;

  std::vector<idpair_t> id_pairs(cloud_in->points.size());

  int i;
// unsigned int idx = 0;
#pragma omp parallel for private(i)  // Multi-thread
  for (i = 0; i < cloud_in->points.size(); i++) {
    unsigned long long vx =
        floor((cloud_in->points[i].x - min_p.coeff(0)) * inverse_voxel_size);
    unsigned long long vy =
        floor((cloud_in->points[i].y - min_p.coeff(1)) * inverse_voxel_size);
    unsigned long long vz =
        floor((cloud_in->points[i].z - min_p.coeff(2)) * inverse_voxel_size);

    unsigned long long voxel_idx = vx * mul_vx + vy * mul_vy + vz * mul_vz;
    idpair_t pair;
    pair.idx = i;
    pair.voxel_idx = voxel_idx;
    // id_pairs.push_back(pair);
    id_pairs[i] = pair;
  }

  // Do sorting
  std::sort(id_pairs.begin(), id_pairs.end());

  int begin_id = 0;

  while (begin_id < id_pairs.size()) {
    cloud_out->push_back(cloud_in->points[id_pairs[begin_id].idx]);

    int compare_id = begin_id + 1;
    while (compare_id < id_pairs.size() &&
           id_pairs[begin_id].voxel_idx == id_pairs[compare_id].voxel_idx)
      compare_id++;
    begin_id = compare_id;
  }

  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used =
      std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

  // free the memory
  std::vector<idpair_t>().swap(id_pairs);

  std::cout << "\n"
            << "[" << cloud_out->points.size()
            << "] points remain after the downsampling in ["
            << time_used.count() * 1000.0 << "] ms." << std::endl;

  return 1;
}

PointCloudXYZI FeatureExtract::getFeature(
    pcl::PointCloud<PointINNO>::Ptr &in_pc) {
  pcl::PointCloud<PointINNO>::Ptr pc_down(new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_ground(new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_ground_down(
      new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_unground(new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_vertex(new pcl::PointCloud<PointINNO>());

  pcl::PointCloud<PointINNO>::Ptr pc_pillar(new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_beam(new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_facade(new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_roof(new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_pillar_down(
      new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_beam_down(
      new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_facade_down(
      new pcl::PointCloud<PointINNO>());
  pcl::PointCloud<PointINNO>::Ptr pc_roof_down(
      new pcl::PointCloud<PointINNO>());

  float vf_downsample_resolution = 0.5;
  int gf_grid_pt_num_thre = 8;
  float gf_grid_resolution = 3.0;
  float gf_max_grid_height_diff = 0.3;
  float gf_neighbor_height_diff = 1.5;
  float gf_max_ground_height = 3.0;  // 3.0
  int gf_down_rate_ground = 2;
  int gf_down_down_rate_ground = 2;
  int gf_downsample_rate_nonground = 3;
  int gf_reliable_neighbor_grid_thre = 0;
  int estimate_ground_normal_method = 3;
  float normal_estimation_radius = 2.0;
  int distance_inverse_sampling_method = 0;
  float standard_distance = 300.0;
  bool fixed_num_downsampling = false;
  int ground_down_fixed_num = 500;
  bool extract_curb_or_not = false;
  float intensity_thre = FLT_MAX;
  bool apply_scanner_filter = false;
  float pca_neighbor_radius = 2.0;  // 0.6

  int pca_neighbor_k = 25;
  int pca_neighbor_k_min = 8;
  int pca_down_rate = 1;
  float edge_thre = 0.65;
  float planar_thre = 0.65;
  float edge_thre_down = 0.75;
  float planar_thre_down = 0.75;
  int extract_vertex_points_method = 0;  // 2 to enable vertex extract
  float curvature_thre = 0.12;
  float linear_vertical_sin_high_thre = 0.94;  // pillar > 70 deg(1.22)  0.94
  float linear_vertical_sin_low_thre = 0.17;   // beam < 10 deg
  float planar_vertical_sin_high_thre = 0.98;
  float planar_vertical_sin_low_thre = 0.34;
  int pillar_down_fixed_num = 200;
  int facade_down_fixed_num = 800;
  int beam_down_fixed_num = 200;
  int roof_down_fixed_num = 200;
  int unground_down_fixed_num = 20000;
  float beam_height_max = FLT_MAX;
  float roof_height_min = 0.0;
  float feature_pts_ratio_guess = 0.3;
  bool sharpen_with_nms_on = false;
  bool use_distance_adaptive_pca = false;

  // if(have_semantic_mask){remove_dynamic_obj;}
  // else  // filter the point cloud of the back of the vehicle itself and the
  // underground

  // voxel_downsample();
  //   voxel_downsample(in_pc, pc_down, vf_downsample_resolution);
  // random_downsample();
  random_downsample_pcl(in_pc, pc_down, 40000);
  std::cout << "input size: " << in_pc->size()
            << "   downsampled size: " << pc_down->size() << std::endl;
  //   pc_down = in_pc;
  // fast_ground_filter();
  // TicToc t_ground;
  fast_ground_filter(
      pc_down, pc_ground, pc_ground_down, pc_unground, pc_vertex,
      gf_grid_pt_num_thre, gf_grid_resolution, gf_max_grid_height_diff,
      gf_neighbor_height_diff, gf_max_ground_height, gf_down_rate_ground,
      gf_down_down_rate_ground, gf_downsample_rate_nonground,
      gf_reliable_neighbor_grid_thre, estimate_ground_normal_method,
      normal_estimation_radius, distance_inverse_sampling_method,
      standard_distance, fixed_num_downsampling, ground_down_fixed_num,
      extract_curb_or_not, intensity_thre, apply_scanner_filter);
  // std::cout << "find ground: " << t_ground.toc() << std::endl;

  float vertex_curvature_non_max_r = 1.5 * pca_neighbor_radius;
  // classify_nground_pts()
  // TicToc t_unground;
  classify_nground_pts(
      pc_unground, pc_pillar, pc_beam, pc_facade, pc_roof, pc_pillar_down,
      pc_beam_down, pc_facade_down, pc_roof_down, pc_vertex,
      pca_neighbor_radius, pca_neighbor_k, pca_neighbor_k_min, pca_down_rate,
      edge_thre, planar_thre, edge_thre_down, planar_thre_down,
      extract_vertex_points_method, curvature_thre, vertex_curvature_non_max_r,
      linear_vertical_sin_high_thre, linear_vertical_sin_low_thre,
      planar_vertical_sin_high_thre, planar_vertical_sin_low_thre,
      fixed_num_downsampling, pillar_down_fixed_num, facade_down_fixed_num,
      beam_down_fixed_num, roof_down_fixed_num, unground_down_fixed_num,
      beam_height_max, roof_height_min, feature_pts_ratio_guess,
      sharpen_with_nms_on, use_distance_adaptive_pca);

  PointCloudXYZI pc_surf;
  for (int i = 0; i < pc_beam->points.size(); i++) {
    pcl::PointXYZINormal point_tmp;
    point_tmp.x = pc_beam->points[i].x;
    point_tmp.y = pc_beam->points[i].y;
    point_tmp.z = pc_beam->points[i].z;
    point_tmp.intensity = pc_beam->points[i].intensity;
    pc_surf.push_back(point_tmp);
  }
  for (int i = 0; i < pc_pillar->points.size(); i++) {
    pcl::PointXYZINormal point_tmp;
    point_tmp.x = pc_pillar->points[i].x;
    point_tmp.y = pc_pillar->points[i].y;
    point_tmp.z = pc_pillar->points[i].z;
    point_tmp.intensity = pc_pillar->points[i].intensity;
    pc_surf.push_back(point_tmp);
  }
  // for (int i = 0; i < pc_vertex->points.size(); i++)
  // {
  //     pcl::PointXYZINormal point_tmp;
  //     point_tmp.x = pc_vertex->points[i].x;
  //     point_tmp.y = pc_vertex->points[i].y;
  //     point_tmp.z = pc_vertex->points[i].z;
  //     point_tmp.intensity = pc_vertex->points[i].intensity;
  //     pc_surf.push_back(point_tmp);
  // }
  for (int i = 0; i < pc_facade->points.size(); i++) {
    pcl::PointXYZINormal point_tmp;
    point_tmp.x = pc_facade->points[i].x;
    point_tmp.y = pc_facade->points[i].y;
    point_tmp.z = pc_facade->points[i].z;
    point_tmp.intensity = pc_facade->points[i].intensity;
    pc_surf.push_back(point_tmp);
  }
  return pc_surf;

  // if (pc_surf.size() != 0)
  // {
  //     return pc_surf;
  // }
  // else
  // {
  //     return;
  // }

  // update the parameters according to the situation
  // std::cout << "find unground: " << t_unground.toc() << std::endl;

  // if (pc_ground->size() != 0)
  // {
  //     data_io.saveFeaturePCD(*pc_ground,
  //     "/home/xng/catkin_ws/src/feature_test/data/mulls_ground.pcd");
  // }
  // if (pc_unground->size() != 0)
  // {
  //     data_io.saveFeaturePCD(*pc_unground,
  //     "/home/xng/catkin_ws/src/feature_test/data/mulls_unground.pcd");
  // }
  // if (pc_pillar->size() != 0)
  // {
  //     data_io.saveFeaturePCD(*pc_pillar,
  //     "/home/xng/catkin_ws/src/feature_test/data/mulls_pillar.pcd");
  // }
  // if (pc_beam->size() != 0)
  // {
  //     data_io.saveFeaturePCD(*pc_beam,
  //     "/home/xng/catkin_ws/src/feature_test/data/mulls_beam.pcd");
  // }
  // if (pc_facade->size() != 0)
  // {
  //     data_io.saveFeaturePCD(*pc_facade,
  //     "/home/xng/catkin_ws/src/feature_test/data/mulls_facade.pcd");
  // }
  // if (pc_vertex->size() != 0)
  // {
  //     data_io.saveFeaturePCD(*pc_vertex,
  //     "/home/xng/catkin_ws/src/feature_test/data/mulls_vertex.pcd");
  // }

  // data_io.saveFeaturePCD(*pc_roof,
  // "/home/xng/catkin_ws/src/feature_test/data/mulls_roof.pcd");
  // data_io.saveFeaturePCD(*pc_pillar_down,
  // "/home/xng/catkin_ws/src/feature_test/data/mulls_pillar_down.pcd");
  // data_io.saveFeaturePCD(*pc_beam_down,
  // "/home/xng/catkin_ws/src/feature_test/data/mulls_beam_down.pcd");
  // data_io.saveFeaturePCD(*pc_facade_down,
  // "/home/xng/catkin_ws/src/feature_test/data/mulls_facade_down.pcd");
  // data_io.saveFeaturePCD(*pc_roof_down,
  // "/home/xng/catkin_ws/src/feature_test/data/mulls_roof_down.pcd");

  // return true;
}
