#pragma once

#include <utils/yaml_utils.h>
#include <iostream>

namespace ctrlvio {

const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 2000;

// For Pointgrey camera. Euroc dataset is 460.0;
extern double FOCAL_LENGTH;

extern double INIT_DEPTH;
extern double MIN_PARALLAX;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern double GRAVITY_NORM;

void readParameters(const YAML::Node& node);

}  // namespace ctrlvio
