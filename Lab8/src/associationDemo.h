#ifndef ASSOCIATION_DEMO_H
#define ASSOCIATION_DEMO_H

#include "cameraModel.hpp"
#include "settings.h"

void runDescriptorMatcher(const Settings & s, const CameraParameters & param, bool doCalibrationGridInFill);
void runCompatibleDescriptorMatcher(const Settings & s, const CameraParameters & param, bool doCalibrationGridInFill);
void runGeometricMatcher(const Settings & s, const CameraParameters & param, bool doCalibrationGridInFill);


#endif