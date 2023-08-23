#include "parameters.h"
#include <opencv2/core.hpp>

#include <utils/opt_weight.h>

namespace ctrlvio {

double INIT_DEPTH;
double MIN_PARALLAX;

double FOCAL_LENGTH;

double ACC_N = 8.0e-2;
double GYR_N = 4.0e-3;
double ACC_W = 4.0e-5;
double GYR_W = 2.0e-6;
double GRAVITY_NORM = 9.8;

void readParameters(const YAML::Node& node) {
  /// IMU parameters
  IMUNoise imu_noise(node);
  ACC_N = imu_noise.sigma_a;
  GYR_N = imu_noise.sigma_w;
  ACC_W = imu_noise.sigma_ab;
  GYR_W = imu_noise.sigma_wb;

  std::string config_path = node["config_path"].as<std::string>();
  std::string imu_yaml = config_path + node["imu_yaml"].as<std::string>();
  YAML::Node imu_node = YAML::LoadFile(imu_yaml);
  GRAVITY_NORM = yaml::GetValue<double>(imu_node, "gravity_mag", 9.80);

  /// camera parameters
  std::string cam_yaml = config_path + node["cam_yaml"].as<std::string>();
  cv::FileStorage fsSettings(cam_yaml, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }

  FOCAL_LENGTH = fsSettings["focal_length"];

  MIN_PARALLAX = fsSettings["keyframe_parallax"];
  MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

  INIT_DEPTH = 5.0;

  fsSettings.release();
}

}  // namespace ctrlvio
