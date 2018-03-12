/*
*  detector.h
*  
*  Landing target detection for MBZIRC competition
*  
*  Created on: 7. 1. 2016
*  Author: petr stepan
*/


#ifndef __CROSS_DETECTOR_H__
#define __CROSS_DETECTOR_H__

#include <opencv2/opencv.hpp>

class exposure_balance {
public:
  int wh_satur;
  int bl_satur;
  int wh_trg_sat;
  int bl_trg_sat;
  int wh_lvl;
  int bl_lvl;
  int exposure_value;
  int change_exp;
};

int detect_cr(cv::Mat src, cv::Mat *img, exposure_balance *eb, cv::RotatedRect &ell, bool use_gui, float &distance);
void detect_init(void);
void set_camera_param(cv::Mat intr, cv::Mat dist, float dx, float dy);

#endif
