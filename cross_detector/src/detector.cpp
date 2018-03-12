/*
*  detector.cpp
*  
*  Landing target detection for MBZIRC competition
*  
*  Created on: 7. 1. 2016
*  Author: petr stepan
*/
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cstdio>
#ifdef _DEBUG_DETECTOR
#include "detector.h"
#else
#include <cross_detector/detector.h>
#endif
#include <time.h>

using namespace std;
using namespace cv;


// if you want to see partial results of the detection, you can uncomment some of these _DEBUG definitions
//#define _DEBUG_DETECTOR

//#define _DEBUG_LINES
//#define _DEBUG_CROSS
//#define _DEBUG_CIRCLE
//#define _DEBUG_OUTPUT
//#define _DEBUG_IMG
//#define _DEBUG_SMALL

#define MAX_HEIGHT 1024
#define DETECT_HISTORY 4

#define sqr(a) ((a)*(a))
#define abs(a) (((a)<0)?(-(a)):(a))
#define my_abs(a) (((a)<0)?(-(a)):(a))

static RNG rng(12345);
static int skip;

static int pmin[MAX_HEIGHT];
static int pmax[MAX_HEIGHT];

static float last_ell_size;
static float real_ell_size;
static float best_ell_size;
static float last_line_size;
static int last_detect;
static int last_thresh;
static int last_col_min, last_col_max;
static int last_bl_sat, last_wh_sat;
static RotatedRect last_ell;

static Mat *dbg;
static Mat *src_img_ptr;
static int height, width;

static Mat metric_coordinates;
static Mat image_coordinates;
static Mat intrinsic;
static Mat distCoeffs;
static Mat element_er;
static Mat element_er_cross;
static Mat pic, tmp_pic;
static float cam_cx, cam_cy, cam_fx, cam_fy, cam_scale, cam_dx, cam_dy;

static Mat img2, tmp, tmp2, tmp3, tmp4, tmp_img;
static Mat orig;
static Mat buf;

static Mat element;

static Mat glob_tmp;
static Mat erode_el;
static Mat skel, seg;

// Sizes for adaptive thresholding (see Section 2.1 and Figure 2.)
static int small_box_size = 2; // box size 2*2+1=5 pixels
static int box_size = 5;       // box size 2*5+1=11 pixels
static int big_box_size = 12;  // box size 2*12+1=25 pixels
static int cross_det;

static float *src_ptr, *dst_ptr;

static float glob_dist;


// Precalculate array sigma=>scale_buf (see Section 2.2)
static float scale_buf[2000];
static void omnidir_undistortPoints(float dist_x, float dist_y, float &undist_x, float &undist_y);

static bool simul_camera;

// Precalculate array sigma=>scale_buf (see Section 2.2), and store information about camera
void set_camera_param(Mat intr, Mat dist, float dx, float dy) {
  metric_coordinates = cv::Mat(1, 1, CV_32FC2);
  image_coordinates = cv::Mat(1, 1, CV_32FC2);
  src_ptr = image_coordinates.ptr<float>(0);
  dst_ptr = metric_coordinates.ptr<float>(0);
  intrinsic = intr;
  distCoeffs = dist;
  double k[4];
  k[0] = dist.at<float>(0);
  k[1] = dist.at<float>(1);
  k[2] = dist.at<float>(2);
  k[3] = dist.at<float>(3);
  cam_fx = intr.at<float>(0, 0);
  cam_fy = intr.at<float>(1, 1);
  cam_scale = (cam_fx+cam_fy)/2.0;
  cam_cx = intr.at<float>(0, 2);
  cam_cy = intr.at<float>(1, 2);
  cam_dx = dx;
  cam_dy = dy;

  cout << "CAMERA fx " << cam_fx << " fy " << cam_fy << " cx " << cam_cx << " cy " << cam_cy << " dx " << cam_dx << " dy " << cam_dy << endl;
  cout << "Distortion " << k[0] << ", " << k[1] << ", " << k[2] << ", " << k[3] << endl;

  scale_buf[0]=0;
  for (int i=1; i<2000; i++) {
    double theta_d = (double)i*1.547/2000.0;
    double theta = theta_d;
    for(int j = 0; j < 10; j++ ) {
      double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
      theta = theta_d / (1 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8);
    }
    scale_buf[i] = float(std::tan(theta) / theta_d);
    if (scale_buf[i]>4000.0) {
      scale_buf[i]=4000.0;
    }
  }
}

// optimized undistort function see Section 2.2
static void omnidir_undistortPoints(float dist_x, float dist_y, float &undist_x, float &undist_y)
{
  float pp_x = (dist_x-cam_cx)/cam_fx;
  float pp_y = (dist_y-cam_cy)/cam_fy;
  float scale = 1.0;
  float theta_d = sqrt(pp_x*pp_x + pp_y*pp_y);
    
  int i = (int)(theta_d*(2000.0/1.547)+0.5);
  if (i>=0) {
    if (i<2000) {
      scale = scale_buf[i];
    } else {
      scale=2000.0;
    }
  }
  undist_x = pp_x*cam_fx*scale+cam_cx;
  undist_y = pp_y*cam_fy*scale+cam_cy;
}


// Colours for debug printing
static const Vec3b bcolors[] = { Vec3b(0, 0, 255), Vec3b(0, 255, 0), Vec3b(0, 255, 255), Vec3b(255, 0, 0), Vec3b(255, 128, 0), Vec3b(255, 255, 0),
    Vec3b(0, 128, 255), Vec3b(255, 0, 255), Vec3b(255, 255, 255), Vec3b(120, 120, 120) };
// Function for debug printing
static void cross(Point pp, int c) {
    for (int j = -4; j < 5; j++) {
        Point p = pp;
        p.x += j;
        dbg->at<Vec3b>(p) = bcolors[c % 9];
        p.y += 1;
        dbg->at<Vec3b>(p) = bcolors[c % 9];
        p.y -= 2;
        dbg->at<Vec3b>(p) = bcolors[c % 9];
    }
    for (int j = -4; j < 5; j++) {
        Point p = pp;
        p.y += j;
        dbg->at<Vec3b>(p) = bcolors[c % 9];
        p.x += 1;
        dbg->at<Vec3b>(p) = bcolors[c % 9];
        p.x -= 2;
        dbg->at<Vec3b>(p) = bcolors[c % 9];
    }
}
// Function for debug printing
static void dr_point(Point pp, int c) {
    if (pp.x>=0 && pp.x<=780 && pp.y>-0 && pp.y<480) 
      dbg->at<Vec3b>(pp) = bcolors[c % 9];
}

#define MAX_LINES 200
static vector<Point> lin[MAX_LINES];
static Point2f line_end_a[MAX_LINES];
static Point2f line_end_b[MAX_LINES];
static Vec4f v_line[MAX_LINES];
static float c[MAX_LINES];
static int line_ptr[MAX_LINES];

static Vec4f cen_line[MAX_LINES];
static float cen_line_size[MAX_LINES];
static float line_len[MAX_LINES];
static float line_dist_max[MAX_LINES];
static float line_dist_avg[MAX_LINES];
static int lines;

#define BIG_MAX_LINES 400
static Point2f big_end_a[BIG_MAX_LINES];
static Point2f big_end_b[BIG_MAX_LINES];
static Vec4f big_v_line[BIG_MAX_LINES];
static float big_c[BIG_MAX_LINES];
static int big_line_ptr[BIG_MAX_LINES];
static int big_lines;


static float dist(Point2f &a, Point2f &b) {
    return sqrt(sqr(a.x-b.x) + sqr(a.y-b.y));
}

static float dist(Point &a, Point &b) {
    return sqrt(sqr(a.x-b.x) + sqr(a.y-b.y));
}

static float dist(int l, Point2f &b) {
    return abs(v_line[l][1] * b.x - v_line[l][0] * b.y - c[l]);
}

// intersection of line a and b from array cen_line, the result is stored into point p
// it returns true if intersestion exists
static bool intersection_poc(int a, int b, Point2f &p) {
    float d = cen_line[a][0] * cen_line[b][1] - cen_line[a][1] * cen_line[b][0];
    if (abs(d) < 1e-8) {
        return false;
    } else {
        double t = ((cen_line[b][2] - cen_line[a][2]) * cen_line[b][1] - (cen_line[b][3] - cen_line[a][3]) * cen_line[b][0]) / d;
        p.x = cen_line[a][2] + t * cen_line[a][0];
        p.y = cen_line[a][3] + t * cen_line[a][1];
    }
    return true;
}

// intersection of line a and b from array v_line, the result is stored into point p
// it returns true if intersestion exists
static bool intersection(int a, int b, Point2f &p) {
    float d = v_line[a][0] * v_line[b][1] - v_line[a][1] * v_line[b][0];
    if (abs(d) < 1e-8) {
        return false;
    } else {
        double t = ((v_line[b][2] - v_line[a][2]) * v_line[b][1] - (v_line[b][3] - v_line[a][3]) * v_line[b][0]) / d;
        p.x = v_line[a][2] + t * v_line[a][0];
        p.y = v_line[a][3] + t * v_line[a][1];
    }
    return true;
}

// it finds end points for set of points in array lin
static void end_points(int a, Point2f &aa, Point2f &bb) {
    int min_x, max_x, min_x_p = 0, max_x_p = 0;
    int min_y, max_y, min_y_p = 0, max_y_p = 0;

    min_x = max_x = lin[a][0].x;
    min_y = max_y = lin[a][0].y;
    for (int i = 1; i < lin[a].size(); i++) {
        if (lin[a][i].x < min_x) {
            min_x = lin[a][i].x;
            min_x_p = i;
        }
        if (lin[a][i].x > max_x) {
            max_x = lin[a][i].x;
            max_x_p = i;
        }
        if (lin[a][i].y < min_y) {
            min_y = lin[a][i].y;
            min_y_p = i;
        }
        if (lin[a][i].y > max_y) {
            max_y = lin[a][i].y;
            max_y_p = i;
        }
    }
    if ((max_x - min_x) > (max_y - min_y)) {
        aa.x = lin[a][min_x_p].x;
        aa.y = lin[a][min_x_p].y;
        bb.x = lin[a][max_x_p].x;
        bb.y = lin[a][max_x_p].y;
    } else {
        aa.x = lin[a][min_y_p].x;
        aa.y = lin[a][min_y_p].y;
        bb.x = lin[a][max_y_p].x;
        bb.y = lin[a][max_y_p].y;
    }
}


/**
* Code for thinning a binary image using Guo-Hall algorithm.
*/

/**
* Perform one thinning iteration.
* Normally you wouldn't call this function directly from your code.
*
* @param  im    Binary image with range = 0-1
* @param  iter  0=even, 1=odd
*/
void thinningGuoHallIteration(cv::Mat& im, int iter) {
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows; i++) {
        for (int j = 1; j < im.cols; j++) {
            uchar p2 = im.at<uchar>(i - 1, j);
            uchar p3 = im.at<uchar>(i - 1, j + 1);
            uchar p4 = im.at<uchar>(i, j + 1);
            uchar p5 = im.at<uchar>(i + 1, j + 1);
            uchar p6 = im.at<uchar>(i + 1, j);
            uchar p7 = im.at<uchar>(i + 1, j - 1);
            uchar p8 = im.at<uchar>(i, j - 1);
            uchar p9 = im.at<uchar>(i - 1, j - 1);

            int C = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) + (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N = N1 < N2 ? N1 : N2;
            int m = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

            if (C == 1 && (N >= 2 && N <= 3) & m == 0)
                marker.at<uchar>(i, j) = 1;
        }
    }
    im &= ~marker;
}

static int num_thin;
/**
* Function for thinning the given binary image
*
* @param  im  Binary image with range = 0-255
*/
void thinningGuoHall(cv::Mat& im) {
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    num_thin = 0;
    do {
        thinningGuoHallIteration(im, 0);
        thinningGuoHallIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
        num_thin++;
    } while (cv::countNonZero(diff) > 0 && num_thin < 20);

    im *= 255;
}



// add points to lines after Guo-Hall thinning
// these local variable stores information about lines
static int col_min, col_max;
static int cross_thresh;
static Mat skel_c, thr_i, eroded, tmp_i;

#define MAX_SMALL_LINES 16
static Point last_p[MAX_SMALL_LINES];
static int is_open[MAX_SMALL_LINES];
static int was_used[MAX_SMALL_LINES];
static RotatedRect *undist_ell;

// find which line is the nearest to point (x,y) and add this point to the line
static void add_point_small(int x, int y) {
    int find = -1;
    for (int ii = 0; ii < lines && find < 0; ii++) {
        if (is_open[ii] && abs(last_p[ii].x-x) < 2 && abs(last_p[ii].y-y) < 2) {
            was_used[ii] = 1;
            last_p[ii].x = x;
            last_p[ii].y = y;
            lin[ii].push_back(last_p[ii]);
            line_ptr[ii]++;
            find = ii;
        }
    }
    for (int ii = 0; ii < lines && find < 0; ii++) {
        if (is_open[ii] && abs(last_p[ii].x-x) < 3 && abs(last_p[ii].y-y) < 3) {
            was_used[ii] = 1;
            last_p[ii].x = x;
            last_p[ii].y = y;
            lin[ii].push_back(last_p[ii]);
            line_ptr[ii]++;
            find = ii;
        }
    }
    if (find < 0 && lines < MAX_SMALL_LINES) {
        last_p[lines].x = x;
        last_p[lines].y = y;
        was_used[lines] = 1;
        is_open[lines] = 1;
        lin[lines].push_back(last_p[lines]);
        line_ptr[lines++] = 1;
    }
}


static int stack_fill_x[1000];
static int stack_fill_y[1000];
static int stack_ptr;

// test cross with Guo Hall thinning or segmentation for realy small targets
static bool testSmallCross(int min_y, int max_y, int min_x, int max_x, uchar *p, int width, RotatedRect *ell, InputArray _orig, int thr) {
  double t1 = (double) getTickCount();
  Mat orig = _orig.getMat();
  int line_size_min = (max_x - min_x) * 12 / 200.0 + (max_y - min_y) * 12 / 200.0;
  int x, y;
  bool ret = false;
  int ROI_max_x = (max_x - min_x), ROI_max_y = (max_y - min_y);
  int ROI_cen_x = ROI_max_x/2, ROI_cen_y = ROI_max_y/2;
  int act_box_size;
  
  // select Region Of Interest for detected ellipse
  Mat orig_ROI = orig(Rect(min_x, min_y, ROI_max_x, ROI_max_y));
  if (ROI_max_x>ROI_max_y) {
    act_box_size = 2*(ROI_max_x/12)+1;
  } else {
    act_box_size = 2*(ROI_max_x/12)+1;
  }
  if (act_box_size>21) {
    act_box_size =21;
  } else if (act_box_size<5) {
    act_box_size = 5;
  }
  // prepare adaptive thresholding for selected ROI
  adaptiveThreshold(orig_ROI, thr_i , 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, act_box_size, 1);
  bool done;
  int num = 0;
  
  lines=0;  
  if (ROI_max_x>30 && ROI_max_y>30) {
    // if region is big enough to use Guo-Hall thinning
    int x_poc = ROI_cen_x, y_poc = ROI_cen_y;
    int size = 2, ptr;
    int y1 = y_poc - size, y2 = y_poc + size;
    int x1 = x_poc - size, x2 = x_poc + size;
    
    // perform Guo-Hall thinning
    thinningGuoHall(thr_i);
#ifdef _DEBUG_SMALL
    imshow("thresh", thr_i);
#endif
    // find lines in the result of Guo-Hall thinning
    // initialize the lines from center in four directions
    for (x = x1, ptr = 0; ptr < 2 * size + 1; ptr++, x++) {
      if (thr_i.at<uchar>(y1, x) > 0) {
        add_point_small(x + min_x, y1 + min_y);
#ifdef _DEBUG_SMALL
        dr_point(Point(x+min_x, y1+min_y),7);
      } else {
        dr_point(Point(x+min_x, y1+min_y),6);
#endif
      }
      if (thr_i.at<uchar>(y2, x) > 0) {
        add_point_small(x + min_x, y2 + min_y);
#ifdef _DEBUG_SMALL
        dr_point(Point(x+min_x, y2+min_y),7);
      } else {
        dr_point(Point(x+min_x, y2+min_y),6);
#endif
      }
    }
    for (y = y1 + 1, ptr = 1; ptr < 2 * size; ptr++, y++) {
      if (thr_i.at<uchar>(y, x1) > 0) {
        add_point_small(x1 + min_x, y + min_y);
#ifdef _DEBUG_SMALL
        dr_point(Point(x1+min_x, y+min_y),7);
      } else {
        dr_point(Point(x1+min_x, y+min_y),6);
#endif
      }
      if (thr_i.at<uchar>(y, x2) > 0) {
        add_point_small(x2 + min_x, y + min_y);
#ifdef _DEBUG_SMALL
        dr_point(Point(x2+min_x, y+min_y),7);
      } else {
        dr_point(Point(x2+min_x, y+min_y),6);
#endif
      }
    }
    size++;
    int open_lines = lines;

    // find lines from the center of image; adding points to existing lines or create new one
    // the limit is ellipse size scaled by ellipse border
    while ((size * 1.4142) < (ell->size.height + ell->size.width) / 4.2 && x1 > 0 && x2 < ROI_max_x - 1 && y1 > 0 && y2 < ROI_max_y - 1
     && (lines < 4 || open_lines > 2)) {
      x1--;
      y1--;
      x2++;
      y2++;
      for (x = x_poc; x <= x2; x++) {
        if (thr_i.at<uchar>(y1, x) > 0) {
          add_point_small(x + min_x, y1 + min_y);
#ifdef _DEBUG_SMALL
          dr_point(Point(x+min_x, y1+min_y),7);
        } else {
          dr_point(Point(x+min_x, y1+min_y),6);
#endif
        }
        if (thr_i.at<uchar>(y2, x) > 0) {
          add_point_small(x + min_x, y2 + min_y);
#ifdef _DEBUG_SMALL
          dr_point(Point(x+min_x, y2+min_y),7);
        } else {
          dr_point(Point(x+min_x, y2+min_y),6);
#endif
        }
      }
      for (x = x_poc; x >= x1; x--) {
        if (thr_i.at<uchar>(y1, x) > 0) {
          add_point_small(x + min_x, y1 + min_y);
#ifdef _DEBUG_SMALL
          dr_point(Point(x+min_x, y1+min_y),7);
        } else {
          dr_point(Point(x+min_x, y1+min_y),6);
#endif
        }
        if (thr_i.at<uchar>(y2, x) > 0) {
          add_point_small(x + min_x, y2 + min_y);
#ifdef _DEBUG_SMALL
          dr_point(Point(x+min_x, y2+min_y),7);
        } else {
          dr_point(Point(x+min_x, y2+min_y),6);
#endif
        }
      }
      for (y = y_poc; y < y2; y++) {
        if (thr_i.at<uchar>(y, x1) > 0) {
          add_point_small(x1 + min_x, y + min_y);
#ifdef _DEBUG_SMALL
          dr_point(Point(x1+min_x, y+min_y),7);
        } else {
          dr_point(Point(x1+min_x, y+min_y),6);
#endif
        }
        if (thr_i.at<uchar>(y, x2) > 0) {
          add_point_small(x2 + min_x, y + min_y);
#ifdef _DEBUG_SMALL
          dr_point(Point(x2+min_x, y+min_y),7);
        } else {
          dr_point(Point(x2+min_x, y+min_y),6);
#endif
        }
      }
      for (y = y_poc; y > y1; y--) {
        if (thr_i.at<uchar>(y, x1) > 0) {
          add_point_small(x1 + min_x, y + min_y);
#ifdef _DEBUG_SMALL
          dr_point(Point(x1+min_x, y+min_y),7);
        } else {
          dr_point(Point(x1+min_x, y+min_y),6);
#endif
        }
        if (thr_i.at<uchar>(y, x2) > 0) {
          add_point_small(x2 + min_x, y + min_y);
#ifdef _DEBUG_SMALL
          dr_point(Point(x2+min_x, y+min_y),7);
        } else {
          dr_point(Point(x2+min_x, y+min_y),6);
#endif
        }
      }
      open_lines = 0;
      for (int ii = 0; ii < lines; ii++) {
        if (was_used[ii] == 0) {
          is_open[ii] = 0;
        } else {
          was_used[ii] = 0;
          open_lines++;
        }
      }
      size++;
    }
    // for all lines compute undistortion coordinates
    for (int l1 = 0; l1 < lines; l1++) {
      if (line_ptr[l1] > 2) {
        for (int ii = 0; ii < line_ptr[l1]; ii++) {
          omnidir_undistortPoints(lin[l1][ii].x, lin[l1][ii].y, dst_ptr[0], dst_ptr[1]);
          lin[l1][ii].x = dst_ptr[0];
          lin[l1][ii].y = dst_ptr[1];
        }
        // compute parameters of lines
        fitLine(lin[l1], v_line[l1], CV_DIST_L2, 0, 0.01, 0.01);
        c[l1] = v_line[l1][1] * v_line[l1][2] - v_line[l1][0] * v_line[l1][3];
        end_points(l1, line_end_a[l1], line_end_b[l1]);
      }
    }
    // merge pairs of lines that are parallel and close enough
    for (int l1 = 0; l1 < lines - 1; l1++) {
      if (line_ptr[l1] > 2) {
        for (int l2 = l1 + 1; l2 < lines; l2++) {
          if (line_ptr[l2] > 2) {
            if (v_line[l1][0] * v_line[l2][0] + v_line[l1][1] * v_line[l2][1] > 0.9 && (dist(l1, line_end_a[l2]) < 6 || dist(l1, line_end_b[l2]) < 6)) {
              for (int ii = 0; ii < line_ptr[l2]; ii++) {
                lin[l1].push_back(lin[l2][ii]);
              }
              line_ptr[l1] += line_ptr[l2];
              line_ptr[l2] = 0;
              fitLine(lin[l1], v_line[l1], CV_DIST_L2, 0, 0.01, 0.01);
              c[l1] = v_line[l1][1] * v_line[l1][2] - v_line[l1][0] * v_line[l1][3];
              end_points(l1, line_end_a[l1], line_end_b[l1]);
            }
          }
        }
      }
    }
      
    Point2f pp;
    Point2f fin;
    int max_sum = 0;
    // find intersection of lines that is close to center of ellipse and contains the highest number of points
    for (int l1 = 0; l1 < lines - 1; l1++) {
      if (line_ptr[l1] > 4) {
        for (int l2 = l1 + 1; l2 < lines; l2++) {
          if (line_ptr[l2] > 4) {
            if (intersection(l1, l2, pp)) {
              if (abs(pp.x - ell->center.x) < (max_x - min_x) / 6 && abs(pp.y - ell->center.y) < (max_y - min_y) / 6
               && line_ptr[l1] + line_ptr[l2] > max_sum) {
                max_sum = line_ptr[l1] + line_ptr[l2];
                fin.x = pp.x;
                fin.y = pp.y;
                last_line_size = num_thin / 2.2;
                ret = true;
              }
            }
          }
        }
      }
    }
    if (ret && (max_sum > 12)) {
      cout << "FIND" << ell->center << " novy "<<fin<<" max sum "<< max_sum<< "size " << (max_x-min_x) << ","<< (max_y-min_y)<<endl;
      ell->center.x = fin.x;
      ell->center.y = fin.y;
    } else if (ret) {
      ret = false;
      cout << "max sum pouze " << max_sum << endl;
    }
    double t2 = (double) getTickCount();
    cout << "Small cross detector " << ((t2 - t1) * 1000. / getTickFrequency()) << "ms" << endl;
  } else {

// the ellipse is to small to use adaptive thresholding to find lines
// we are not looking for lines we are detecting areas between cross     

// use morphology to detect regions between cross
    dilate(thr_i, tmp_i, element_er);
    dilate(thr_i, eroded, element_er_cross);
    erode(eroded, thr_i, element_er_cross);
#ifdef _DEBUG_SMALL
    imshow("thresh1", thr_i);
#endif
    int num_area=0;
    // detect areas inside dilated and eroded ROI by filling algorithm
    for (int y=0; y<ROI_max_y; y++) {
      for (int x=1+pmin[y+min_y]-min_x; x<(pmax[y+min_y]-min_x)-1; x++) {
        if (thr_i.at<uchar>(y, x)<100) {
          thr_i.at<uchar>(y, x)=120;
          int area_size=0;
          stack_ptr=1;
          stack_fill_x[0]=x;
          stack_fill_y[0]=y;
          while (stack_ptr>0) {
            stack_ptr--;
            int xx=stack_fill_x[stack_ptr];
            int yy=stack_fill_y[stack_ptr];
            area_size++;
            if (xx>0 && thr_i.at<uchar>(yy, xx-1)<100) {
              thr_i.at<uchar>(yy, xx-1)=120;
              stack_fill_x[stack_ptr]=xx-1;
              stack_fill_y[stack_ptr++]=yy;
            }
            if (xx+1<ROI_max_x && thr_i.at<uchar>(yy, xx+1)<100) {
              thr_i.at<uchar>(yy, xx+1)=120;
              stack_fill_x[stack_ptr]=xx+1;
              stack_fill_y[stack_ptr++]=yy;
            }
            if (yy>0 && thr_i.at<uchar>(yy-1, xx)<100) {
              thr_i.at<uchar>(yy-1, xx)=120;
              stack_fill_x[stack_ptr]=xx;
              stack_fill_y[stack_ptr++]=yy-1;
            }
            if (yy+1<ROI_max_y && thr_i.at<uchar>(yy+1, xx)<100) {
              thr_i.at<uchar>(yy+1, xx)=120;
              stack_fill_x[stack_ptr]=xx;
              stack_fill_y[stack_ptr++]=yy+1;
            }
          }
          if (area_size>2) {
            num_area++;
          }
        }
      }
      
      // if there are 4 areas the cross was detected
      if (num_area==4) {
        ret=true;
      } else {
        // try to erode ones more the image and then detect areas  
        erode(tmp_i, thr_i, element_er_cross);
        num_area=0;
        // detect areas in the ROI
        for (int y=0; y<ROI_max_y; y++) {
          for (int x=1+pmin[y+min_y]-min_x; x<(pmax[y+min_y]-min_x)-1; x++) {
            if (thr_i.at<uchar>(y, x)<100) {
              thr_i.at<uchar>(y, x)=120;
              int area_size=0;
              stack_ptr=1;
              stack_fill_x[0]=x;
              stack_fill_y[0]=y;
              while (stack_ptr>0) {
                stack_ptr--;
                int xx=stack_fill_x[stack_ptr];
                int yy=stack_fill_y[stack_ptr];
                area_size++;
                if (xx>0 && thr_i.at<uchar>(yy, xx-1)<100) {
                  thr_i.at<uchar>(yy, xx-1)=120;
                  stack_fill_x[stack_ptr]=xx-1;
                  stack_fill_y[stack_ptr++]=yy;
                }
                if (xx+1<ROI_max_x && thr_i.at<uchar>(yy, xx+1)<100) {
                  thr_i.at<uchar>(yy, xx+1)=120;
                  stack_fill_x[stack_ptr]=xx+1;
                  stack_fill_y[stack_ptr++]=yy;
                }
                if (yy>0 && thr_i.at<uchar>(yy-1, xx)<100) {
                  thr_i.at<uchar>(yy-1, xx)=120;
                  stack_fill_x[stack_ptr]=xx;
                  stack_fill_y[stack_ptr++]=yy-1;
                }
                if (yy+1<ROI_max_y && thr_i.at<uchar>(yy+1, xx)<100) {
                  thr_i.at<uchar>(yy+1, xx)=120;
                  stack_fill_x[stack_ptr]=xx;
                  stack_fill_y[stack_ptr++]=yy+1;
                }
              }
              if (area_size>2) {
                num_area++;
              }
            }
          }
        }
        // if 4 areas were detected the cross is present inside the ellipse
        if (num_area==4) {
          ret=true;
        }
      }
    }
#ifdef _DEBUG_SMALL
    cout << "Num lines "<<lines;
    imshow("thresh-after", thr_i);
    imshow("dbg", *dbg);
    waitKey(0);
#endif      
  }
  return ret;
}
  
  
  
// --------------------- algorithm for split and merge lines or arcs on the border of objects  ------------------------------
// --------------------- the result of this algorithm is to fill information about lines and arcs into global variables  --------------------  

// direction for border following
//       5 6 7
//        \|/
//       4- -0
//        /|\
//       3 2 1
static int dir_x[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
static int dir_y[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
static int next_dir_clock[8] = { 6, 6, 0, 0, 2, 2, 4, 4 };
static int next_dir_counter[8] = { 2, 4, 4, 6, 6, 0, 0, 2 };
static Point buf_poi[16384];
static Point2f buf_poi_tr[16384];
static struct pair {
    int a, b;
} stack_pair[400];

bool ellipse_found;
static vector<Point> inside_ell;
RotatedRect *ell_glob;

#define _MAX_CIRCLES 100
static int num_circles;
static Point2f circ_center[_MAX_CIRCLES];
static float circ_radius[_MAX_CIRCLES];
static float circ_angle[_MAX_CIRCLES];
static int circ_to_circ[_MAX_CIRCLES];
static int circ_num_point[_MAX_CIRCLES];

// compute center and radius of arc from 3 points p1, p2, p3; p1 and p3 are ends of the arc
inline bool getCircle(Point2f& p1, Point2f& p2, Point2f& p3, Point2f& center, float &radius) {
    bool ret = true;
    float x1 = p1.x;
    float x2 = p2.x;
    float x3 = p3.x;

    float y1 = p1.y;
    float y2 = p2.y;
    float y3 = p3.y;

    center.x = (x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2);
    center.y = (x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1);
    float d = 2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2);
    if (abs(d) > 1e-10) {
        center.x /= d;
        center.y /= d;
        radius = sqrt((center.x - x1) * (center.x - x1) + (center.y - y1) * (center.y - y1));
        ret = (radius < 1000.0);
    } else {
        ret = false;
    }

    return ret;
}


// the main function that follow the border of the object clokwise of counter-clockwise and detect lines and arc on the border
static int get_lines(int x, int y, uchar *o, uchar *s, int min_x, int max_x, int min_y, int max_y, int width, int thresh, int dir, bool clock, int &x_end, int &y_end,
        int &dir_end, int limit_dist) {
// (x,y) is staring point on the border
// o - is the image
// s - is the array that define if this point was already used
// min_x, max_x, min_y, max_y - borders of the image
// width length of one line
// thresh the value for detecting obejct in the image
// dir - initial direction to follow the border
// clock - clockwise or counterclockwise border following
// x_end, y_end, dir_end - output for next border detection
// limit dist - parameter for line merging
            
    int ptr = x + y * width;
    int next_ptr;
    int buf_ptr_poi = 0;
    int skel_repeat = 0;
    int old_lines = lines;
    RotatedRect inside_rect;
    bool new_ell = false;

    // find the direction for border following
    while (skel_repeat < 3 && x > min_x && x < max_x - 1 && y > min_y && y < max_y - 1) {
        int d;
        if (clock) {
            d = 1;
        } else {
            d = 7;
        }
        while (d != 0 && d != 8) {
            next_ptr = ptr + dir_x[(dir + d) % 8] + width * dir_y[(dir + d) % 8];
            if (o[next_ptr] > thresh) {
                if (s[next_ptr] > 0) {
                    skel_repeat++;
                } else {
                    skel_repeat = 0;
                }
                if (skel_repeat < 3) {
                    s[ptr] = 200;
                    ptr = next_ptr;
                    buf_poi[buf_ptr_poi].x = x;
                    buf_poi[buf_ptr_poi++].y = y;
                    x = x + dir_x[(dir + d) % 8];
                    y = y + dir_y[(dir + d) % 8];
                    if (clock) {
                        dir = next_dir_clock[(dir + d) % 8];
                    } else {
                        dir = next_dir_counter[(dir + d) % 8];
                    }
                }
                break;
            }
            if (clock) {
                d++;
            } else {
                d--;
            }
        }
        if (d == 8 || d == 0) {
            skel_repeat = 3;
        }
    }
    
    // create list of points on the object border
    if ((x == min_x || x == max_x - 1 || y == min_y || y == max_y - 1) && buf_ptr_poi > 1) {
        if (x == min_x) {
            if (clock && (dir == 6 || dir == 4)) {
                dir = 2;
            } else if (!clock && (dir == 2 || dir == 4)) {
                dir = 6;
            }
        }
        if (x == max_x - 1) {
            if (clock && (dir == 0 || dir == 2)) {
                dir = 6;
            } else if (!clock && (dir == 0 || dir == 6)) {
                dir = 2;
            }
        }
        if (y == min_y) {
            if (clock && (dir == 2 || dir == 4)) {
                dir = 0;
            } else if (!clock && (dir == 2 || dir == 0)) {
                dir = 4;
            }
        }
        if (y == max_y - 1) {
            if (clock && (dir == 0 || dir == 6)) {
                dir = 4;
            } else if (!clock && (dir == 4 || dir == 6)) {
                dir = 0;
            }
        }
        int iter = 0;
        while (o[next_ptr] > thresh && iter < 2500 && (x == min_x || x == max_x - 1 || y == min_y || y == max_y - 1)) {
            iter++;
            next_ptr = ptr + dir_x[dir] + width * dir_y[dir];
            s[ptr] = 200;
            ptr = next_ptr;
            x = x + dir_x[dir];
            y = y + dir_y[dir];
            if (o[next_ptr] > thresh) {
                if (x == min_x && ((dir == 2 && (y == max_y - 1)) || (dir == 6 && (y == min_y)))) {
                    dir = 4;
                }
                if (x == max_x - 1 && ((dir == 2 && (y == max_y - 1)) || (dir == 6 && (y == min_y)))) {
                    dir = 0;
                }
                if (y == min_y && ((dir == 4 && (x == max_x - 1)) || (dir == 0 && (x == min_x)))) {
                    dir = 2;
                }
                if (y == max_y - 1 && ((dir == 4 && (x == max_x - 1)) || (dir == 0 && (x == min_x)))) {
                    dir = 6;
                }
            }
        }
        x = x - dir_x[dir];
        y = y - dir_y[dir];
        if (x == min_x) {
            x++;
            ptr = x + y * width;
            while (o[ptr] < thresh && iter < 3000 && y + dir_y[(dir + 4) % 8] > min_y && y + dir_y[(dir + 4) % 8] < max_y - 1) {
                iter++;
                y += dir_y[(dir + 4) % 8];
                ptr = ptr + width * dir_y[(dir + 4) % 8];
            }
            while (o[ptr + width * dir_y[dir]] > thresh && iter < 3000 && y + dir_y[dir] > min_y && y + dir_y[dir] < max_y - 1) {
                iter++;
                y += dir_y[dir];
                ptr = ptr + width * dir_y[dir];
            }
        }
        if (x == max_x - 1) {
            x--;
            ptr = x + y * width;
            while (o[ptr] < thresh && iter < 3000 && y + dir_y[(dir + 4) % 8] > min_y && y + dir_y[(dir + 4) % 8] < max_y - 1) {
                iter++;
                y += dir_y[(dir + 4) % 8];
                ptr = ptr + width * dir_y[(dir + 4) % 8];
            }
            while (o[ptr + width * dir_y[dir]] > thresh && iter < 3000 && y + dir_y[dir] > min_y && y + dir_y[dir] < max_y - 1) {
                iter++;
                y += dir_y[dir];
                ptr = ptr + width * dir_y[dir];
            }
        }
        if (y == min_y) {
            y++;
            ptr = x + y * width;
            while (o[ptr] < thresh && iter < 3000 && x + dir_x[(dir + 4) % 8] > min_x && x + dir_x[(dir + 4) % 8] < max_x - 1) {
                iter++;
                x += dir_x[(dir + 4) % 8];
                ptr = ptr + dir_x[(dir + 4) % 8];
            }
            while (o[ptr + dir_x[dir]] > thresh && iter < 3000 && x + dir_x[dir] > min_x && x + dir_x[dir] < max_x - 1) {
                iter++;
                x += dir_x[dir];
                ptr = ptr + dir_x[dir];
            }
        }
        if (y == max_y - 1) {
            y--;
            ptr = x + y * width;
            while (o[ptr] < thresh && iter < 3000 && x + dir_x[(dir + 4) % 8] > min_x && x + dir_x[(dir + 4) % 8] < max_x - 1) {
                iter++;
                x += dir_x[(dir + 4) % 8];
                ptr = ptr + dir_x[(dir + 4) % 8];
            }
            while (o[ptr + dir_x[dir]] > thresh && iter < 3000 && x + dir_x[dir] > min_x && x + dir_x[dir] < max_x - 1) {
                iter++;
                x += dir_x[dir];
                ptr = ptr + dir_x[dir];
            }
        }
        x_end = x;
        y_end = y;
        dir_end = dir;
    } else {
        buf_ptr_poi--;
        if (buf_ptr_poi > 1) {
            x_end = buf_poi[buf_ptr_poi - 1].x;
            y_end = buf_poi[buf_ptr_poi - 1].y;
            dir_end = dir;
        } else { 
            x_end = x;
            y_end = y;
            dir_end = dir;
        }
    }
    if (buf_ptr_poi > 40) {
    // the object is big enough to detect lines and arcs
        int i;
        for (i = 0; i < buf_ptr_poi; i++) {
          // compute undistored coordinates for all points  
          omnidir_undistortPoints(buf_poi[i].x, buf_poi[i].y, buf_poi_tr[i].x, buf_poi_tr[i].y);
        }
        if (!ellipse_found) {
            // first try to detect that the whole border is arc
            for (i = 0; i < buf_ptr_poi; i++) {
                inside_ell.push_back(buf_poi_tr[i]);
            }
            inside_rect = fitEllipse(inside_ell);
            RotatedRect rr = minAreaRect(inside_ell);
            if (dist(ell_glob->center, inside_rect.center) < 10 && inside_rect.size.height * 1.04 > rr.size.height
                    && inside_rect.size.width * 1.04 > rr.size.width) {
                new_ell = true;
                ellipse_found = true;
            }
        }
        if (!new_ell) {
            int max_i = 1;
            int stack_ptr;
            float dist_max = dist(buf_poi_tr[0], buf_poi_tr[1]), dist_tmp;
            // initialize split and merge algorithm
            for (i = 2; i < buf_ptr_poi; i++) {
                dist_tmp = dist(buf_poi_tr[0], buf_poi_tr[i]);
                if (dist_tmp > dist_max) {
                    dist_max = dist_tmp;
                    max_i = i;
                }
            }
            if (max_i < buf_ptr_poi - 1) {
                stack_ptr = 2;
                stack_pair[0].a = 0;
                stack_pair[0].b = max_i;
                stack_pair[1].a = max_i + 1;
                stack_pair[1].b = buf_ptr_poi - 1;
            } else {
                stack_ptr = 1;
                stack_pair[0].a = 0;
                stack_pair[0].b = max_i;
            }
            // there is some line test split and merge 
            while (stack_ptr > 0) {
                bool is_circle=false;
                stack_ptr--;
                int save_a = stack_pair[stack_ptr].a, save_b = stack_pair[stack_ptr].b;
                float a = (buf_poi_tr[save_a].y - buf_poi_tr[save_b].y);
                float b = (buf_poi_tr[save_b].x - buf_poi_tr[save_a].x);
                float d = sqrt(a * a + b * b);
                float max_dist = 0;
                a = a / d;
                b = b / d;
                float cc = a * buf_poi_tr[save_a].x + b * buf_poi_tr[save_a].y;
                float c2 = -a * buf_poi_tr[save_a].y + b * buf_poi_tr[save_a].x;
                bool ok = true;
                for (i = save_a + 1; i < save_b; i++) {
                    float dd = my_abs(a*buf_poi_tr[i].x + b*buf_poi_tr[i].y-cc);
                    float d2 = -a * buf_poi_tr[i].y + b * buf_poi_tr[i].x - c2;
                    if (dd > max_dist) {
                        max_dist = dd;
                        max_i = i;
                    }
                    if (d2 < 0 && d2 > d) {
                        ok = false;
                    }
                }
                float glob_dif = max_dist;
                float max_dif = max_dist;
                float koef_dist = limit_dist;
                if (d>120.0) {
                  koef_dist = limit_dist*d/120.0;
                }
                if (max_dist > koef_dist && (save_b + 1 - save_a) > 20) {
                    // test if the segment is arc
                    Point2f center, center2, center3;
                    float radius, limit_h, limit_l;
                    int num = 0;
                    float angle = 0.0, min_r = 1000000000.0, max_r = 0;
                    float min2_r = 1000000000.0, max2_r = 0;
                    float min3_r = 1000000000.0, max3_r = 0;
                    if (getCircle(buf_poi_tr[save_a], buf_poi_tr[(save_a + save_b) / 2], buf_poi_tr[save_b], center, radius)) {
                        angle = abs((buf_poi_tr[save_a].x-center.x)*(buf_poi_tr[save_b].x-center.x)+(buf_poi_tr[save_a].y-center.y)*(buf_poi_tr[save_b].y-center.y))
                                / (sqrt(sqr(buf_poi_tr[save_a].x-center.x) + sqr(buf_poi_tr[save_a].y-center.y))
                                        * sqrt(sqr(buf_poi_tr[save_b].x-center.x) + sqr(buf_poi_tr[save_b].y-center.y)));
                        center2.x = (buf_poi_tr[(save_a + save_b) / 2].x+1.1*(center.x - buf_poi_tr[(save_a + save_b) / 2].x));
                        center2.y = (buf_poi_tr[(save_a + save_b) / 2].y+1.1*(center.y - buf_poi_tr[(save_a + save_b) / 2].y));
                        center3.x = (buf_poi_tr[(save_a + save_b) / 2].x+1.2*(center.x - buf_poi_tr[(save_a + save_b) / 2].x));
                        center3.y = (buf_poi_tr[(save_a + save_b) / 2].y+1.2*(center.y - buf_poi_tr[(save_a + save_b) / 2].y));
                        if (radius>150) {
                            limit_h = sqr(radius*1.05);
                            limit_l = sqr(radius*0.95);
                        } else {
                            limit_h = sqr(radius*1.09);
                            limit_l = sqr(radius*0.91);
                        }
                        for (i = save_a; i < save_b; i++) {
                            float dd = sqr(buf_poi_tr[i].x-center.x) + sqr(buf_poi_tr[i].y-center.y);
                            float dd2 = sqr(buf_poi_tr[i].x-center2.x) + sqr(buf_poi_tr[i].y-center2.y);
                            float dd3 = sqr(buf_poi_tr[i].x-center3.x) + sqr(buf_poi_tr[i].y-center3.y);
                            if (dd < limit_h && dd > limit_l) {
                                num++;
                            }
                            if (dd > max_r) {
                                max_r = dd;
                            }
                            if (dd < min_r) {
                                min_r = dd;
                            }
                            if (dd2 > max2_r) {
                                max2_r = dd2;
                            }
                            if (dd2 < min2_r) {
                                min2_r = dd2;
                            }
                            if (dd3 > max3_r) {
                                max3_r = dd3;
                            }
                            if (dd3 < min3_r) {
                                min3_r = dd3;
                            }
                        }
                        float roz = sqrt(max_r)-sqrt(min_r);
                        float roz2 = sqrt(max2_r)-sqrt(min2_r);
                        float roz3 = sqrt(max3_r)-sqrt(min3_r);
                        if (num > ((save_b - save_a) * 0.95)) {
                          if (roz2<roz || roz3<roz) {
                            if (roz2<roz3) {
                              center=center2;
                              radius=sqrt(min2_r)+roz2/2.0;
                            } else {
                              center=center3;
                              radius=sqrt(min3_r)+roz3/2.0;
                            }
                          }
                        }
#ifdef _DEBUG_CIRCLE
                        cout << "Circle angle "<< angle<<" radius "<<radius<<" min "<<sqrt(min_r)<<" max "<<sqrt(max_r)<<" num "<<num<<" poc "<<(save_b - save_a)<<" min2 "<<min2_r<<" max2 "<<max2_r<<" radius min2 "<<sqrt(min2_r)<<" radius max2 "<<sqrt(max2_r);
                        cout << " radius min3 "<<sqrt(min3_r)<<" radius max3 "<<sqrt(max3_r)<<" roz "<<roz<<" roz2 "<<roz2<<" roz3 "<<roz3<<endl;
                        line(*dbg, Point(buf_poi_tr[save_a].x* 0.3 + 263,buf_poi_tr[save_a].y * 0.3 + 168), Point(buf_poi_tr[save_b].x* 0.3 + 263,buf_poi_tr[save_b].y * 0.3 + 168), Scalar(255,0,255), 1);
                        imshow("dbg", *dbg);
                        //waitKey(0);
#endif
                    }
                    if ((num > ((save_b - save_a) * 0.95)) && (angle < 0.45 || (angle<0.8 && num>90))) {
#ifdef _DEBUG_CIRCLE
                        circle(*dbg, Point(center.x*0.3+263, center.y*0.3+168), radius*0.3, Scalar(0, 0, 255), 1);
                        cout << "CIRCLE " << num << "," << (save_b - save_a) << " center " << center << " radius " << radius << " angle " << angle << " min " << min_r
                                << " max " << max_r << endl;
                        imshow("dbg", *dbg);
                        waitKey(0);
#endif
                        if (num_circles < _MAX_CIRCLES) {
                            circ_center[num_circles] = center;
                            circ_radius[num_circles] = radius;
                            circ_num_point[num_circles] = num;
                            circ_angle[num_circles++] = angle;
                        } else if (angle < 0.2) {
                            for (int ii = 0; ii < num_circles; ii++) {
                                if (circ_angle[ii] > angle && circ_num_point[ii] < num) {
                                    circ_center[ii] = center;
                                    circ_radius[ii] = radius;
                                    circ_num_point[ii] = num;
                                    circ_angle[ii] = angle;
                                }
                            }
                        }
                        glob_dif = 0;
                        max_dif = 0;
                        is_circle=true;
                    }
                }
                if (!is_circle) {    
                    // it is not arc try to split and merge
                    if (max_dist > koef_dist) {
                        if ((max_i - save_a) > (limit_dist + 1) && stack_ptr < 380) { // 7
                            stack_pair[stack_ptr++].b = max_i;
                        }
                        if ((save_b - max_i) > (limit_dist + 1) && stack_ptr < 380) { // 8
                            stack_pair[stack_ptr].a = max_i + 1;
                            stack_pair[stack_ptr++].b = save_b;
                        }
                    } else {
                        if (save_b - save_a - 1 > (limit_dist + 1) && lines < MAX_LINES - 1) {  // limit dist is here like limit in number of points
                            lin[lines].clear();
                            for (i = save_a; i <= save_b; i++) {
                                lin[lines].push_back(buf_poi_tr[i]);
                            }
                            fitLine(lin[lines], v_line[lines], CV_DIST_L2, 0, 0.01, 0.01);
                            c[lines] = v_line[lines][1] * v_line[lines][2] - v_line[lines][0] * v_line[lines][3];
                            line_ptr[lines] = lin[lines].size();
                            lines++;
                        }
                    }
                }
            }
        }
    }
    return buf_ptr_poi;
}


// -------------------------- test the cross inside the circle ---------------------------------    ^   
// -------------------------- if the ellipse is smaller than 145x145 - call function testSmallCross |
// -------------------------- if the ellipse is big - use function get lines to detect lines inside the ellipse
//
static bool testCross(int min_y, int max_y, int min_x, int max_x, uchar *p, int width, RotatedRect *ell, InputArray _orig, bool small) {
    Mat orig = _orig.getMat();
    uchar *o = orig.ptr(0);
    uchar *s;
    bool ret = false;
    int x, y;
    int ptr;
    int tmp_max, tmp_min;
    long int suma;
    int thresh = 0, cnt = 0;
    int zac_x, zac_y;
    int limit_x = (max_x - min_x) / 6;
    int limit_y = (max_y - min_y) / 6;
    int line_size_min = (max_x - min_x) * 10 / 200.0 + (max_y - min_y) * 10 / 200.0;
    int line_size_max = (max_x - min_x) * 40 / 200.0 + (max_y - min_y) * 40 / 200.0;
    int end_x, end_y, end_dir;
    if (small) {
        limit_x = limit_y = 2;
    }

    lines = 0;
    for (int i = 0; i < MAX_LINES; i++) {
        lin[i].clear();
    }

    col_min = 255;
    col_max = 0;
    for (y = min_y + 1; y < max_y - 1; y++) {
        ptr = pmin[y] + y * width;
        suma = 0;
        for (x = pmin[y] + 2; x < pmax[y] - 1; x++) {
            suma += o[ptr];
            if (o[ptr] > col_max){
                col_max = o[ptr];
            }
            if (o[ptr] < col_min) {
                col_min = o[ptr];
            }
            ptr++;
        }
        if (pmax[y] - pmin[y] + 1 + cnt > 0.1) {
            thresh = (thresh * cnt + suma) / (pmax[y] - pmin[y] + 1 + cnt);
        }
        cnt += (pmax[y] - pmin[y] + 1);
    }
    cross_thresh = thresh = (2 * col_max + col_min) / 3;
    if ((max_x - min_x)*(max_y - min_y) < 21025) {  
    // the ellipse area is less 145*145 - we can use Gua Hall thinning or segement detection
        ret = testSmallCross(min_y, max_y, min_x, max_x, p, width, ell, _orig, thresh);
    } else {
        // Gua hall thinning will be too slow we need detect lines
#ifdef _DEBUG_CROSS
        cout << "test cross od " << min_y << " do " << max_y << " thresh " << thresh << endl;
        cout << "test cross size " << (max_y - min_y) << ", " << (max_x - min_x) << " line min " << line_size_min << " line max " << line_size_max << endl;
        for (y = min_y + 1; y < max_y - 1; y++) {
            ptr = pmin[y] + y * width;
            suma = 0;
            for (x = pmin[y] + 2; x < pmax[y] - 1; x++) {
                if (o[ptr++] < thresh) {
                    dbg->at<Vec3b>(Point(x, y)) = Vec3b(0, 0, 0);
                } else {
                    dbg->at<Vec3b>(Point(x, y)) = Vec3b(255, 255, 255);
                }
            }
        }
#endif
        skel_c = Mat::zeros(orig.size(), CV_8UC1);
        ellipse_found = false;
        ell_glob = ell;
        inside_ell.reserve(1000);
        inside_ell.clear();
        s = skel_c.ptr(0);
        for (y = min_y + 1 + line_size_min; y < (max_y - 1) - line_size_min; y++) {
            ptr = pmin[y] + line_size_min + y * width;
            x = pmin[y] + line_size_min;
            while (x < (pmax[y] - 1) - line_size_min && !ret) {
                if (o[ptr] < thresh && o[ptr + 1] > thresh && s[ptr] == 0) {
                    get_lines(x + 1, y, o, s, min_x, max_x, min_y, max_y, width, thresh, 0, true, end_x, end_y, end_dir, 3);
                }
                if (o[ptr] > thresh && o[ptr + 1] < thresh && s[ptr + 1] == 0) {
                    get_lines(x, y, o, s, min_x, max_x, min_y, max_y, width, thresh, 4, false, end_x, end_y, end_dir, 3);
                }
                x++;
                ptr++;
            }
        }
#ifdef _DEBUG_CROSS
	cout << "celkem lines " << lines << endl;
        for (int l1 = 0; l1 < lines - 1; l1++) {
          cout << "velikost " << l1 << " je " << line_ptr[l1] << endl;
          line(*dbg, Point(cvRound(v_line[l1][2]-200*v_line[l1][0]), cvRound(v_line[l1][3]-200*v_line[l1][1])),
            Point(cvRound(v_line[l1][2] + 200 * v_line[l1][0]), cvRound(v_line[l1][3] + 200 * v_line[l1][1])),
            Scalar(bcolors[1 + l1%8][0],bcolors[1 + l1%8][1],bcolors[1 + l1%8][2]), 2);
        }
#endif
        Point2f pp;
        Point2f fin;
        int max_sum = 0;
        int poc = 0;
        for (int l1 = 0; l1 < lines; l1++) {
            end_points(l1, line_end_a[l1], line_end_b[l1]);
        }

        // merge lines inside the whole ellipse
        for (int l1 = 0; l1 < lines - 1; l1++) {
            if (line_ptr[l1] > 4) {
                float c_l1 = v_line[l1][1] * v_line[l1][2] - v_line[l1][0] * v_line[l1][3];
                for (int l2 = l1 + 1; l2 < lines; l2++) {
                    if (line_ptr[l2] > 4 && abs(v_line[l1][0] * v_line[l2][0] + v_line[l1][1] * v_line[l2][1]) > 0.96
                            && (abs(v_line[l1][1] * v_line[l2][2] - v_line[l1][0] * v_line[l2][3]-c[l1]) < line_size_min)) {
                        for (int ii = 0; ii < line_ptr[l2]; ii++) {
                            lin[l1].push_back(lin[l2][ii]);
                        }
                        line_ptr[l1] += line_ptr[l2];
                        line_ptr[l2] = 0;
                        fitLine(lin[l1], v_line[l1], CV_DIST_L2, 0, 0.01, 0.01);
                        c[l1] = v_line[l1][1] * v_line[l1][2] - v_line[l1][0] * v_line[l1][3];
                        end_points(l1, line_end_a[l1], line_end_b[l1]);
                    }
                }
            }
        }

#ifdef _DEBUG_CROSS
        cout << "after connection " << lines << endl;
        for (int l1 = 0; l1 < lines; l1++) {
            if (line_ptr[l1]>4)
            line(*dbg, Point(line_end_a[l1].x, line_end_a[l1].y), Point(line_end_b[l1].x, line_end_b[l1].y),
                    Scalar(bcolors[1 + l1 % 8][0], bcolors[1 + l1 % 8][1], bcolors[1 + l1 % 8][2]), 2);
        }
#endif
        // merge lines inside the whole ellipse
        for (int l1 = 0; l1 < lines - 1; l1++) {
            if (line_ptr[l1] > 4) {
                float len_l1 = dist(line_end_a[l1], line_end_b[l1]);
                if (len_l1 < line_size_max) {
                    continue;
                }
                for (int l2 = l1 + 1; l2 < lines; l2++) {
                    if (line_ptr[l2] > 4) {
                        float len_l2 = dist(line_end_a[l2], line_end_b[l2]);
                        float dist_l1l2 = abs(v_line[l1][1] * v_line[l2][2] - v_line[l1][0] * v_line[l2][3]-c[l1]);
                        if (abs(v_line[l1][0] * v_line[l2][0] + v_line[l1][1] * v_line[l2][1]) > 0.9) {
                            if (dist_l1l2 < line_size_max && dist_l1l2 > line_size_min && len_l2 >= line_size_max) {
                                cen_line_size[poc] = dist_l1l2;
                                cen_line[poc][0] = v_line[l1][0];
                                cen_line[poc][1] = v_line[l1][1];
                                cen_line[poc][2] = (v_line[l1][2] + v_line[l2][2]) / 2.0;
                                cen_line[poc][3] = (v_line[l1][3] + v_line[l2][3]) / 2.0;
                                poc++;
                            }
                        }
                    }
                }
            }
        }
        
        // compute lines intersection and detect intersetion near to center of ellipse
        for (int l1 = 0; l1 < poc - 1; l1++) {
            for (int l2 = l1 + 1; l2 < poc; l2++) {
                if (intersection_poc(l1, l2, pp)) {
                    if (abs(pp.x - ell->center.x) < limit_x / 2.0 && abs(pp.y - ell->center.y) < limit_y / 2.0 && line_ptr[l1] + line_ptr[l2] > max_sum) {
                        max_sum = line_ptr[l1] + line_ptr[l2];
                        fin.x = pp.x;
                        fin.y = pp.y;
                        last_line_size = (cen_line_size[l1] + cen_line_size[l2]) / 2.0;
                        ret = true;
                    }
                }
            }
        }
        // if the number of points that create lines with intersetion near ellipse center is big enough
        // we find the cross
        if (ret && ((small && max_sum > 15) || (!small && max_sum > 30))) {
            ell->center.x = fin.x;
            ell->center.y = fin.y;
        } else if (ret) {
            ret = false;
#ifdef _DEBUG_CROSS
            cout << "max sum only " << max_sum << endl;
#endif
        }

#ifdef _DEBUG_CROSS
        cout << "Cross "<< ell->center<<endl;
        imshow("test", *dbg);
#endif
    }
    return ret;
}

#define LIMIT_PIX 1
static vector<Point> point_ellipse;
static vector<Point> point_ellipse_tr;
static RotatedRect ell_tr;


// --------------------- find border of convex area -------------------------------------------
// --------------------- test if it si ellipse ------------------------------------------------
// --------------------- for detected ellipses try to detect cross ----------------------------
static int testCircle(int x, int y, uchar *p, InputArray _orig, RotatedRect &ell, bool small) {
    Mat orig = _orig.getMat();
    Size pic_size = orig.size();
    height = pic_size.height;
    width = pic_size.width;
    int i;
    int ptr = x + y * width, y_ptr;
    int next_min, next_max;
    int min_r, max_l, lim;
    int min_y = y, max_y = y;
    int min_x, max_x;
    int size = 1;
    bool last_line = false;
    bool ret = false;
    int xx;
    int limit_fail;
    int value = -1;
    int orig_y = y;
    RotatedRect und_ell;
    Vec3b color2 = Vec3b(255, 255, 0);
    Vec3b color = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255),
            rng.uniform(0, 255));
#ifdef _DEBUG_CIRCLE
    cout << "Start test circle " << x << "," << y << endl;
#endif
    for (i = 0; i < MAX_HEIGHT; i++) {
        pmin[i] = pmax[i] = -1;
    }
    pmin[y] = x++;
    p[ptr++] = 128;
    while (x < width && p[ptr] < 10) {
#ifdef _DEBUG_OUTPUT
        dbg->at<Vec3b>(Point(x, y)) = color;
#endif
        size++;
        x++;
        p[ptr++] = 128;
    };
    pmax[y] = x - 1;
    // pmin[y] and pmax[y] contains left and right border of the area for all lines y
    max_x = min_r = pmax[y];
    min_x = max_l = pmin[y];
    next_min = pmin[y];
    next_max = pmax[y];
    y++;
    y_ptr = y * width;
#ifdef _DEBUG_CIRCLE
    cout << "Test min " << pmin[y - 1] << " max " << pmax[y - 1] << endl;
#endif

    // line scanning for convex border 
    // left border - min_x, right border max_x

    while (y < height) {
        ptr = y_ptr + next_min;
        // move the left border
        if (p[ptr] < 130 || p[ptr - 1] < 130) {
            while (next_min > 0 && p[ptr - 1] < 130) {
                next_min--;
                ptr--;
            }
        } else {
            while (next_min <= min_r && p[ptr] >= 130) {
                next_min++;
                ptr++;
            }
            if (p[ptr] >= 130) {
                last_line = true;
                max_y = y - 1;
                ptr = pmin[max_y] + width * max_y;
                for (int xx = pmin[max_y]; xx < pmax[max_y]; xx++) {
                    if (p[ptr] > 140) {
                        last_line = false;
                    }
                }
                if (last_line) {
                    size += pmax[max_y] - pmin[max_y] - 1;
                }
                break;
            }
        }

        // move the right border
        ptr = y_ptr + next_max;
        if (p[ptr] < 130 || p[ptr + 1] < 130) {
            while (next_max < (width - 1) && p[ptr + 1] < 130) {
                next_max++;
                ptr++;
            }
        } else {
            while (next_max >= max_l && p[ptr] >= 130) {
                next_max--;
                ptr--;
            }
            if (p[ptr] >= 130) {
                max_y = y - 1;
                break;
            }
        }
        xx = next_min;
        ptr = y_ptr + xx;
        lim = 0;
        while (lim < LIMIT_PIX && xx < next_max) {
            if (p[ptr] >= 130) {
                lim++;
            } else {
                min_r = xx;
            }
            ptr++;
            xx++;
        }

        // test end of the area 
        // and change image to new values
        xx = next_max;
        ptr = y_ptr + xx;
        lim = 0;
        while (lim < LIMIT_PIX && xx > next_min) {
            if (p[ptr] >= 130) {
                lim++;
            } else {
                max_l = xx;
            }
            ptr--;
            xx--;
        }

        xx = next_min;
        ptr = y_ptr + xx;
        while (xx <= min_r) {
            if (p[ptr] >= 130) {
                p[ptr] = 128;
            } else {
                p[ptr] = 128;
            }
#ifdef _DEBUG_OUTPUT
            dbg->at<Vec3b>(Point(xx, y)) = color;
#endif
            ptr++;
            xx++;
        }
        xx = max_l;
        ptr = y_ptr + xx;
        while (xx <= next_max) {
            if (p[ptr] >= 10) {
                p[ptr] = 128;
            } else {
                p[ptr] = 128;
            }
#ifdef _DEBUG_OUTPUT
            dbg->at<Vec3b>(Point(xx, y)) = color;
#endif
            ptr++;
            xx++;
        }

        // store values and prepare for next line
        pmin[y] = next_min;
        pmax[y++] = next_max;
        if (min_x > next_min) {
            min_x = next_min;
        }
        if (max_x < next_max) {
            max_x = next_max;
        }
        y_ptr += width;
        size += 2;
    }
#ifdef _DEBUG_CIRCLE
    cout << "Size " << size << " y min " << min_y << " y max " << max_y << " small " << small << " last line "<< last_line<< " size x " << (max_x-min_x) << " y " << (max_y-min_y)<<endl;
    imshow("dbg",*dbg);
    if (!skip) {
       if ((waitKey(0)&0xff)=='s') {
          skip=1;
       }
    }
#endif
    // if the area is big enough -- there are two modes small ellipse with area more than 20 pixels, big area with 40 pixels
    if (((small && size > 20) || (!small && size > 40)) && (max_y - min_y) > 9 && (max_x - min_x) > 9 && ((max_y - min_y) > 10 || (max_x - min_x) > 10) && last_line) {
        int fail_sz = 0;

        point_ellipse.reserve(size);
        point_ellipse.clear();
        point_ellipse_tr.reserve(size);
        point_ellipse_tr.clear();
        float dst_ptr[16];
        float min_x_tr=1000000.0;
        float max_x_tr=-1000000.0;
        float min_y_tr=1000000.0;
        float max_y_tr=-1000000.0;
        
        for (int xx = pmin[min_y]; xx <= pmax[min_y]; xx++) {
            // compute undistorted coordinates for points at the top of area
            omnidir_undistortPoints(xx, min_y, dst_ptr[0], dst_ptr[1]);
            
            point_ellipse_tr.push_back(Point(dst_ptr[0], dst_ptr[1]));
            if (dst_ptr[0]<min_x_tr) {
              min_x_tr=dst_ptr[0];
            }
            if (dst_ptr[0]>max_x_tr) {
              max_x_tr=dst_ptr[0];
            }
            if (dst_ptr[1]<min_y_tr) {
              min_y_tr=dst_ptr[1];
            }
            if (dst_ptr[1]>max_y_tr) {
              max_y_tr=dst_ptr[1];
            }
            point_ellipse.push_back(Point(xx, min_y));
        }
        for (int yy = min_y + 1; yy <= max_y; yy++) {
            point_ellipse.push_back(Point(pmin[yy], yy));
            // compute undistorted coordinates for points at the left border of the area
            omnidir_undistortPoints(pmin[yy], yy, dst_ptr[0], dst_ptr[1]);
            point_ellipse_tr.push_back(Point(dst_ptr[0], dst_ptr[1]));
            if (dst_ptr[0]<min_x_tr) {
              min_x_tr=dst_ptr[0];
            }
            if (dst_ptr[0]>max_x_tr) {
              max_x_tr=dst_ptr[0];
            }
            if (dst_ptr[1]<min_y_tr) {
              min_y_tr=dst_ptr[1];
            }
            if (dst_ptr[1]>max_y_tr) {
              max_y_tr=dst_ptr[1];
            }
            point_ellipse.push_back(Point(pmax[yy], yy));
            // compute undistorted coordinates for points at the right border of the area
            omnidir_undistortPoints(pmax[yy], yy, dst_ptr[0], dst_ptr[1]);
            point_ellipse_tr.push_back(Point(dst_ptr[0], dst_ptr[1]));
            if (dst_ptr[0]<min_x_tr) {
              min_x_tr=dst_ptr[0];
            }
            if (dst_ptr[0]>max_x_tr) {
              max_x_tr=dst_ptr[0];
            }
            if (dst_ptr[1]<min_y_tr) {
              min_y_tr=dst_ptr[1];
            }
            if (dst_ptr[1]>max_y_tr) {
              max_y_tr=dst_ptr[1];
            }
        }
        if (last_line) {
            for (int xx = pmin[max_y] + 1; xx <= (pmax[max_y] - 1); xx++) {
            // compute undistorted coordinates for points at the bottom of the area
                omnidir_undistortPoints(xx, max_y, dst_ptr[0], dst_ptr[1]);
                point_ellipse.push_back(Point(xx, max_y));
                point_ellipse_tr.push_back(Point(dst_ptr[0], dst_ptr[1]));
                if (dst_ptr[0]<min_x_tr) {
                    min_x_tr=dst_ptr[0];
                }
                if (dst_ptr[0]>max_x_tr) {
                    max_x_tr=dst_ptr[0];
                }
                if (dst_ptr[1]<min_y_tr) {
                    min_y_tr=dst_ptr[1];
                }
                if (dst_ptr[1]>max_y_tr) {
                    max_y_tr=dst_ptr[1];
                }
            }
        }
        // use the cv::fitEllipse to find ellipse parameters
        und_ell = ell = fitEllipse(point_ellipse_tr);
        undist_ell = &und_ell;
        double cos_a = 1, sin_a = 0, koef = 1;
        double limit;
        double limit_max_x = min_x_tr + (max_x_tr - min_x_tr) * 3.0 / 4.0;
        double limit_min_x = min_x_tr + (max_x_tr - min_x_tr) / 4.0;
        double limit_max_y = min_y_tr + (max_y_tr - min_y_tr) * 3.0 / 4.0;
        double limit_min_y = min_y_tr + (max_y_tr - min_y_tr) / 4.0;
        double min_koef = (ell.size.width <= ell.size.height) ? ell.size.width / ell.size.height : ell.size.height / ell.size.width;
        double size_limit = 2*(((max_x_tr - min_x_tr)>(max_y_tr - min_y_tr))?(max_x_tr - min_x_tr):(max_y_tr - min_y_tr));

#ifdef _DEBUG_CIRCLE
        cout << "LIMITS "<<ell.size.width<<" limit "<<size_limit<<" hei "<< ell.size.height <<" SQR "<<(sqr(ell.center.x-cam_cx)+sqr(ell.center.y-cam_cy))<<" cen x "<< ell.center.x <<" min x "<<limit_min_x <<" max x "<< limit_max_x <<" cen y "<< ell.center.y<< " min y "<< limit_min_y << " max y "<<limit_max_y<< endl;
        if ((max_x-min_x)>70 && (max_y-min_y)>70) {
          for (int aa=0; aa<point_ellipse_tr.size(); aa++) {
             dr_point(Point(point_ellipse_tr[aa].x*0.30+263, point_ellipse_tr[aa].y*0.3+168), 1);      
          }
          imshow("dbg",*dbg);
          if (!skip) {
            if ((waitKey(0)&0xff)=='s') {
              skip=1;
            }
          }
        }
#endif        
        if (ell.size.width < size_limit && ell.size.height < size_limit && ell.center.x > limit_min_x && ell.center.x < limit_max_x && ell.center.y > limit_min_y
                && ell.center.y < limit_max_y && last_line) {
            // if the detected ellipse is correct (center is inside of the area with correct sizes)
            sin_a = sin(ell.angle * M_PI / 180);
            cos_a = cos(ell.angle * M_PI / 180);
            koef = ell.size.width / ell.size.height;
            limit = sqrt(ell.size.width * ell.size.width + ell.size.height * ell.size.height) / 50.0;  // orig 60
            if (sqr(ell.center.x-cam_cx)+sqr(ell.center.y-cam_cy)>120000.0) {
              if (limit < 4.5) {
                limit=4.5;
              }
            } else {
              if (limit < 3.5) {  // orig 2.5
                 limit = 3.5;
              }
            }
            // compute distance of border points to ellipse
            // see Section 2.3 equation 1 and 2
            Mat rot = (Mat_<double>(2, 2) << cos(ell.angle), sin(ell.angle), -sin(ell.angle), cos(ell.angle));
            for (int j = 0; j < (int) point_ellipse_tr.size(); j++) {
                // compute distance of border points to ellipse
                // see Section 2.3 equation 1 and 2
                Point2f vec;
                vec.x = point_ellipse_tr[j].x - ell.center.x;
                vec.y = point_ellipse_tr[j].y - ell.center.y;
                Point2f rot_p;
                rot_p.x = vec.x * cos_a + vec.y * sin_a;
                rot_p.y = (-vec.x * sin_a + vec.y * cos_a) * koef;
                double siz = 2.0 * sqrt(rot_p.x * rot_p.x + rot_p.y * rot_p.y) - ell.size.width;
                if (siz < -limit || siz > limit) {
                    fail_sz++;
                }
            }
            if (size / 30 < 3) {
                limit_fail = 3;
            } else {
                limit_fail = size / 30;
            }
#ifdef _DEBUG_CIRCLE
            cout << "LIMITS "<<ell.size.width<<" <width "<<width<<" hei "<< ell.size.height <<" SQR "<<(sqr(ell.center.x-cam_cx)+sqr(ell.center.y-cam_cy))<<" cen x "<< ell.center.x <<" min x "<<limit_min_x <<" max x "<< limit_max_x <<" cen y "<< ell.center.y<< " min y "<< limit_min_y << " max y "<<limit_max_y<< endl;
            cout << "Fail sz "<< fail_sz << " limit fail "<< limit_fail<< " num points "<<point_ellipse_tr.size()<<" min koef "<<min_koef<<endl;
            ellipse(*dbg, ell, Scalar(0, 255, 0), 1);
            if ((max_x - min_x) > 4 && (max_y - min_y) > 4 && min_koef>0.7) {
              //cout << "Bod "<<point_ellipse[0]<< " undist "<<point_ellipse_tr[0]<<endl;
              for (int j = 0; j < (int) point_ellipse_tr.size(); j++) {
                if (point_ellipse_tr[j].x>=0 && point_ellipse_tr[j].x<752 && point_ellipse_tr[j].y>=0 && point_ellipse_tr[j].y<480) {
                    dbg->at<Vec3b>(point_ellipse_tr[j]) = color;
                }
              }
            }
            imshow("dbg",*dbg);
            if (!skip) {
              if ((waitKey(0)&0xff)=='s') {
                skip=1;
              }
            }
#endif

            if ((min_koef>0.65) && (fail_sz <= limit_fail) && (max_x - min_x) > 10 && (max_y - min_y) > 10 && testCross(min_y, max_y, min_x, max_x, p, width, &ell, _orig, small)) {
                 // the area is correct ellipse and test Cross funtion detect the cross inside
                ret = true;
                float max_c, min_c;
                float a = (ell.center.y-cam_cy);
                float b = (cam_cx - ell.center.x);
                float d = sqrt(a*a+b*b);
                if (my_abs(d)>1e-10) {
                  a/=d;
                  b/=d;
                  max_c=min_c = a*ell.center.x+b*ell.center.y;
                  for (int j = 0; j < (int) point_ellipse_tr.size(); j++) {
                    float c = a*point_ellipse_tr[j].x+b*point_ellipse_tr[j].y;
                    if (c<min_c) {
                      min_c=c;
                    }
                    if (c>max_c) {
                      max_c=c;
                    }
                  }
                } else {
                  max_c=min_c=0;
                }
                real_ell_size = max_c-min_c;
                float ell_qual = ell.size.height / ell.size.width;
                float ell_siz = real_ell_size;
                if (ell.size.width < ell.size.height) {
                    ell_qual = ell.size.width / ell.size.height;
                    ell_siz = ell.size.height;
                }
                // compute score for the ellipse to find the best score in the image
                value = 1 + fail_sz * 20 + (1.1 - ell_qual) * 50;
                if (last_detect < DETECT_HISTORY) {
                    float diff = 0.0;
                    if ((ell_siz - last_ell_size) > last_detect * 0.15 * last_ell_size) {
                        value += 50.0 * (ell_siz - last_ell_size) / last_ell_size;
                    }
                    if ((diff = sqrt(sqr(ell.center.x - last_ell.center.x) + sqr(ell.center.y-last_ell.center.y))) > last_detect * last_ell_size) {
                        value += 50.0 * diff / last_ell_size;
                    }
                }
#ifdef _DEBUG_CIRCLE
                cout << "Ellipse center " << ell.center << " size " << ell.size << "fail " << fail_sz << " z " << size << endl;
                ellipse(*dbg, ell, Scalar(255, 0, 0), 2);
                for (int j = 0; j < (int) point_ellipse_tr.size(); j++) {
                    dr_point(point_ellipse_tr[j], 4);
                    Point2f vec;
                    vec.x = point_ellipse_tr[j].x - ell.center.x;
                    vec.y = point_ellipse_tr[j].y - ell.center.y;
                    Point2f rot_p;
                    rot_p.x = vec.x * cos_a + vec.y * sin_a;
                    rot_p.y = (-vec.x * sin_a + vec.y * cos_a) * koef;
                    double siz = 2.0 * sqrt(rot_p.x * rot_p.x + rot_p.y * rot_p.y) - ell.size.width;
                    double kk = ell.size.width / (2.0 * sqrt(rot_p.x * rot_p.x + rot_p.y * rot_p.y));
                    rot_p.x *= kk;
                    rot_p.y *= kk / koef;
                    vec.x = rot_p.x * cos_a - rot_p.y * sin_a;
                    vec.y = rot_p.x * sin_a + rot_p.y * cos_a;
                    if (siz < -limit || siz > limit) {
                        dr_point(vec + ell.center, 5);
                    } else {
                        dr_point(vec + ell.center, 2);
                    }
                }
#endif
            }
        }
    }

    return value;
}

// -------------------- find target inside image after thresholding ---------------------
// -------------------- find circle (ellipse) and test the presence of the cross
static bool findTarget(OutputArray _pic, InputArray orig, RotatedRect &ell, bool small) {
    bool ret = false;
    RotatedRect tmp_ell;
    int best_val = 100000;
    int best_thresh, best_col_min, best_col_max;
    int val;
    pic = _pic.getMat();
    Size size = pic.size();

    if (pic.isContinuous()) {
        int x, y, next_line = size.width;
        uchar *p = pic.ptr(0);
        int p_ptr = 0;
        for (y = 0; y < size.height && !ret; y++) {
            for (x = 0; x < size.width && !ret; x++) {
                if (p[p_ptr] < 10) {
                    if ((val = testCircle(x, y, p, orig, tmp_ell, small)) >= 0) {
                        // find the ellipse with the best score
                        if (val < best_val) {
                            ell = tmp_ell;
                            best_val = val;
                            best_thresh = cross_thresh;
                            best_col_min = col_min;
                            best_col_max = col_max;
                            best_ell_size = real_ell_size;
                        }
                    }
                }
                p_ptr++;
            }
        }
    }
    if (best_val < 400) {
        // the ellipse is not good enough - cancle false positive
        last_detect = 1;
        last_ell_size = best_ell_size;
        last_thresh = best_thresh;
        last_col_min = best_col_min;
        last_col_max = best_col_max;
    }
    return (best_val < 400);
}



//  --------------------- Detection of Big Cross and Parts of circle ----------------------------------
//  --------------------- Section 2.4. ---------------------------------------------------------------

static float touch_limit;
static int near_touch(int l1, int l2) {
  float c_min,c_max, c_a,c_b;
  int ret=0;
  c_a = (v_line[l1][1] * line_end_a[l2].x - v_line[l1][0] * line_end_a[l2].y - c[l1]);
  c_b = (v_line[l1][1] * line_end_b[l2].x - v_line[l1][0] * line_end_b[l2].y - c[l1]);
  if (c_a<c_b) {
    c_min = c_a;
    c_max = c_b;
  } else {
    c_min = c_b;
    c_max = c_a;
  }
  ret = (c_min<=touch_limit) && (c_max>=-touch_limit);

  c_a = (v_line[l2][1] * line_end_a[l1].x - v_line[l2][0] * line_end_a[l1].y - c[l2]);
  c_b = (v_line[l2][1] * line_end_b[l1].x - v_line[l2][0] * line_end_b[l1].y - c[l2]);
  if (c_a<c_b) {
    c_min = c_a;
    c_max = c_b;
  } else {
    c_min = c_b;
    c_max = c_a;
  }
  ret = ret && (c_min<=touch_limit) && (c_max>=-touch_limit);
  
  return ret;
}

#define _CROSS_NUM 60
static int direction[_CROSS_NUM][_CROSS_NUM];
static int num_direct[_CROSS_NUM];
static int poc_center;

static int cross_number;
static Point2f cross_find[_CROSS_NUM];
static float cross_line_size[_CROSS_NUM];
static float cross_num_point[_CROSS_NUM];
static int cross_circles[_CROSS_NUM];

static int cen_num_point[MAX_LINES];
static int cen_min_ind[MAX_LINES];
static int cross_touch[MAX_LINES];

static int add_center_line(int d, float expected, float min_exp, float max_exp, int *list) {
  float val[_CROSS_NUM];
  int l = direction[d][0];
  int ret = 0;
  int list_ptr=0;
  for (int i = 0; i < num_direct[d]; i++) {
    val[i] = (v_line[l][1] * v_line[direction[d][i]][2] - v_line[l][0] * v_line[direction[d][i]][3] - c[l]);
  }
  float min_dist = 100000000.0;
  float best_dist = 1000.0;
  int ind_min = -1;
  int ind_max = -1;
  for (int i = 0; i < num_direct[d] - 1; i++) {
    float c_min,c_max, c_a,c_b;
    int l1 = direction[d][i];
    c_a = line_end_a[l1].x * v_line[l1][0] + line_end_a[l1].y * v_line[l1][1];
    c_b = line_end_b[l1].x * v_line[l1][0] + line_end_b[l1].y * v_line[l1][1];
    if (c_a<c_b) {
      c_min = c_a;
      c_max = c_b;
    } else {
      c_min = c_b;
      c_max = c_a;
    }
    for (int j = i+1; j<num_direct[d]; j++) {
      int l2 = direction[d][j];
      c_a = line_end_a[l2].x * v_line[l1][0] + line_end_a[l2].y * v_line[l1][1];
      c_b = line_end_b[l2].x * v_line[l1][0] + line_end_b[l2].y * v_line[l1][1];
      if (!((c_a<c_min && c_b<c_min)||(c_a>c_max && c_b>c_max))) {
        float dist = my_abs(val[j] - val[i]);
        if (dist<=max_exp && dist>=min_exp && list_ptr<48) {
          list[list_ptr++]=l1;
          list[list_ptr++]=l2;
        }
        if (my_abs(dist-expected)<min_dist ) {
          min_dist = my_abs(dist-expected);
          best_dist = dist;
          ind_min = i;
          ind_max = j;
        }
      }
    }    
  }
  if (ind_min >= 0 && best_dist<=max_exp && best_dist>=min_exp) {
    int tmp;
    cen_line[d][0] = v_line[direction[d][ind_min]][0];
    cen_line[d][1] = v_line[direction[d][ind_min]][1];
    cen_line[d][2] = (v_line[direction[d][ind_min]][2] + v_line[direction[d][ind_max]][2]) / 2.0;
    cen_line[d][3] = (v_line[direction[d][ind_min]][3] + v_line[direction[d][ind_max]][3]) / 2.0;
    cen_line_size[d] = best_dist;
    cen_num_point[d] = line_ptr[direction[d][ind_min]]+line_ptr[direction[d][ind_max]];
    cen_min_ind[d]=0;
    tmp = direction[d][ind_min];
    direction[d][ind_min] = direction[d][0];
    direction[d][0] = tmp;
    tmp = direction[d][ind_max];
    direction[d][ind_max] = direction[d][1];
    direction[d][1] = tmp;
    ret = 1;
  }
  return (list_ptr/2);
}



#define PIXEL_MIN_DIFF 18
// find cross and part of the circle
// the cross is intersection of two pairs of parralel lines
// if part of the circle is detected it supports the correct target detection
static bool findBigCross(OutputArray _pic, InputArray _orig, RotatedRect &ell) {
    bool ret = false;
    RotatedRect tmp_ell;
    Mat pic = _pic.getMat();
    Mat orig = _orig.getMat();
    Size size = orig.size();
    Point2f pp;
    Point2f fin;
    int max_sum = 0;
    float limit, max_limit;
    float min_line_len, min_line_sqr;
    float exp_line_width, min_exp, max_exp;
    
    cross_number = 0;
    num_circles = 0;
    skel = Mat::zeros(orig.size(), CV_8UC1);
    poc_center = 0;
    // set expected width of the cross - distance of two parallel lines
    // the expected width depends on previous detection or on the UAV's altitude
    if (last_detect < DETECT_HISTORY) {
        limit = last_line_size * 0.5;
        max_limit = last_line_size * 1.5;
    } else {
        if (glob_dist>0.1) {
           limit = 0.3 * 492.0 * 0.156 / glob_dist;
           max_limit = 1.5 * 492.0 * 0.156 / glob_dist; 
        } else {
           limit = 0.3 * 492.0 * 0.156 / 0.1;
           max_limit = 1.5 * 492.0 * 0.156 / 0.1;
        }
    }
    if (glob_dist>=0.14) {
        min_line_len = 0.08 * cam_scale / glob_dist;
        exp_line_width = 0.1* cam_scale / glob_dist;
        limit = 0.02* cam_scale / glob_dist;
        max_limit = 3*limit;
    } else {
        min_line_len = 0.08 * cam_scale / 0.14;
        exp_line_width = 0.1* cam_scale / 0.14;
        limit = 0.02* cam_scale / 0.14;
        max_limit = 3*limit;
    }
    if (glob_dist>0.59) {
      max_exp=0.1* cam_scale / (glob_dist-0.5);
    } else {
      max_exp=0.1*cam_scale / 0.09;
    }
    if (glob_dist+0.5>0.15) {
      min_exp=0.1* cam_scale / (glob_dist+0.5);
    } else {
      min_exp=0.1* cam_scale / 0.15;
    }
    touch_limit = limit;
    min_line_sqr = min_line_len*min_line_len;
    // all detected lines are stored in global array, that describe the lines parameters
    lines = 0;
    big_lines=0;
    for (int i = 0; i < MAX_LINES; i++) {
        lin[i].clear();
        line_ptr[i] = 0;
    }
    if (orig.isContinuous()) {
        int x, y, next_line = size.width;
        uchar *o = orig.ptr(0);
        int p_ptr = 0;
        int num, x_start, y_start, x_end, y_end, dir_end;

        // use addpative threshold with box size 25 pixels
        adaptiveThreshold(orig, pic, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 2 * big_box_size + 1, 1);
        // remove the noice in the picture by closing operation
        dilate(pic, tmp_pic, element_er_cross);
        erode(tmp_pic, pic, element_er_cross);
        
        uchar *p = pic.ptr(0);
        uchar *s = skel.ptr(0);
        
        int fill_255=0;
        int fill_0=0;
        int ptr=0;
        // set up the border of the image to make continuous detection of objects borders
        for (y=0;y<2;y++) {
          ptr=y*size.width*(size.height-1)+1;
          for (x=1; x<size.width-2; x++) {
            if (p[ptr]>220 && p[ptr+1]<220 && fill_0==0) {
              if (my_abs(o[ptr-1]-o[ptr+2])<PIXEL_MIN_DIFF) {
                fill_255=1;
              }
            } else if (p[ptr]<220 && p[ptr+1]>220 && fill_255==0) {
              if (my_abs(o[ptr-1]-o[ptr+2])<PIXEL_MIN_DIFF) {
                fill_0=1;
              }
            } else if (fill_255 && p[ptr]<220) {
              p[ptr]=255;
            } else if (fill_0 && p[ptr]>220) {
              p[ptr]=0;
            } else {
              fill_255=0;
              fill_0=0;
            }
            ptr++;
          }
        }

        ellipse_found = true;
        inside_ell.clear();
        // find lines by following border of detected objects and use spilt and merge algorithm for line detection
        // detect only line with predefined minimal length
        for (y = 1; y < size.height - 1; y++) {
            ptr = y * size.width + 1;
            int iteration = 0;
            x = 1;
            while (x < size.width - 1) {
                if (p[ptr] < 220 && p[ptr + 1] > 220 && s[ptr] == 0 && x < size.width - 2 && (o[ptr-1]-o[ptr+2])>PIXEL_MIN_DIFF) {
                    // find the first point on border of new object - detection clockwise
                    x_start = x + 1;
                    y_start = y;
                    // find all line on border of this object
                    num = get_lines(x + 1, y, p, s, 0, size.width, 0, size.height, size.width, 220, 0, true, x_end, y_end, dir_end, 10);
                    while (((abs(x_end-x_start) > 2 || abs(y_end-y_start) > 2) && num > 4) && iteration < 60) {
                        num = get_lines(x_end, y_end, p, s, 0, size.width, 0, size.height, size.width, 220, dir_end, true, x_end, y_end, dir_end, 10);
                        iteration++;
                    }
                }
                if (p[ptr] > 220 && p[ptr + 1] < 220 && s[ptr + 1] == 0 && x < size.width - 2 && (o[ptr+2]-o[ptr-1])>PIXEL_MIN_DIFF) {
                    // find the first point on border of new object - detection counter clockwise
                    x_start = x;
                    y_start = y;
                    // find all line on border of this object
                    num = get_lines(x, y, p, s, 0, size.width, 0, size.height, size.width, 220, 4, false, x_end, y_end, dir_end, 10);
                    while (((abs(x_end-x_start) > 2 || abs(y_end-y_start) > 2) && num > 4) && iteration < 60) {
                        num = get_lines(x_end, y_end, p, s, 0, size.width, 0, size.height, size.width, 220, dir_end, false, x_end, y_end, dir_end, 10);
                        iteration++;
                    }
                }
                s[ptr]=1;
                // analyze all lines from one object
                // merge lines that are near and parallel to create big lines
                for (int l1 = 0; l1 < lines; l1++) {
                    end_points(l1, line_end_a[l1], line_end_b[l1]);
                    if (sqr(line_end_a[l1].x-line_end_b[l1].x)+sqr(line_end_a[l1].y-line_end_b[l1].y) > min_line_len*min_line_len) {
                        float c_min,c_max, c_a,c_b;
                        c_a = line_end_a[l1].x * v_line[l1][0] + line_end_a[l1].y * v_line[l1][1];
                        c_b = line_end_b[l1].x * v_line[l1][0] + line_end_b[l1].y * v_line[l1][1];
                        if (c_a<c_b) {
                          c_min = c_a;
                          c_max = c_b;
                        } else {
                          c_min = c_b;
                          c_max = c_a;
                        }
                        if (line_end_a[l1].x<-10000.0 || line_end_a[l1].y<-10000.0 || line_end_a[l1].x>10000.0 || line_end_a[l1].y>10000.0) {
                           line_ptr[l1]=0;
                           lin[l1].clear();
                        } else {
#ifdef _DEBUG_LINES
                          line(*dbg, Point(line_end_a[l1].x * 0.3 + 263, line_end_a[l1].y * 0.3 + 168),
                          Point(line_end_b[l1].x * 0.3 + 263, line_end_b[l1].y * 0.3 + 168), Scalar(bcolors[8][0], bcolors[8][1], bcolors[8][2]), 1); 
#endif
                          for (int l2 = l1 + 1; l2 < lines; l2++) {
                            if (abs(v_line[l1][0] * v_line[l2][0] + v_line[l1][1] * v_line[l2][1]) > 0.97
                                    && (abs(v_line[l1][1] * v_line[l2][2] - v_line[l1][0] * v_line[l2][3]-c[l1]) < limit) 
                                    && (abs(v_line[l2][1] * v_line[l1][2] - v_line[l2][0] * v_line[l1][3]-c[l2]) < limit)) {
                                end_points(l2, line_end_a[l2], line_end_b[l2]);
                                c_a = line_end_a[l2].x * v_line[l1][0] + line_end_a[l2].y * v_line[l1][1];
                                c_b = line_end_b[l2].x * v_line[l1][0] + line_end_b[l2].y * v_line[l1][1];
                                if ((c_a<c_min && c_b<c_min)||(c_a>c_max && c_b>c_max)) {
                                  for (int ii = 0; ii < lin[l2].size(); ii++) {
                                      lin[l1].push_back(lin[l2][ii]);
                                  }
                                  line_ptr[l1] += lin[l2].size();
                                  line_ptr[l2] = 0;
                                  lin[l2].clear();
                                  fitLine(lin[l1], v_line[l1], CV_DIST_L2, 0, 0.01, 0.01);
                                  c[l1] = v_line[l1][1] * v_line[l1][2] - v_line[l1][0] * v_line[l1][3];
                                  end_points(l1, line_end_a[l1], line_end_b[l1]);
#ifdef _DEBUG_LINES
                                  line(*dbg, Point(line_end_a[l1].x * 0.3 + 263, line_end_a[l1].y * 0.3 + 168),
                                    Point(line_end_b[l1].x * 0.3 + 263, line_end_b[l1].y * 0.3 + 168), Scalar(bcolors[3][0], bcolors[3][1], bcolors[3][2]), 1);
#endif
                                  
                                }
                            }
                          }
                        }
                    }
                }
                // find pairs of parallel lines that can form the cross
                int num_dir = 0;
                for (int l1 = 0; l1 < lines; l1++) {
                  float len_sqr = sqr(line_end_a[l1].x-line_end_b[l1].x)+sqr(line_end_a[l1].y-line_end_b[l1].y);
                  if (len_sqr>min_line_sqr && len_sqr<900*min_line_sqr) {
                        bool find = false;
                        for (int d = 0; d < num_dir && !find; d++) {
                            if (abs(v_line[l1][0] * v_line[direction[d][0]][0] + v_line[l1][1] * v_line[direction[d][0]][1]) > 0.98 && num_direct[d]<_CROSS_NUM-1) {
                                find = true;
                                direction[d][num_direct[d]++] = l1;
#ifdef _DEBUG_LINES
                                line(*dbg, Point(line_end_a[l1].x * 0.3 + 263, line_end_a[l1].y * 0.3 + 168),
                                  Point(line_end_b[l1].x * 0.3 + 263, line_end_b[l1].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 2); 
#endif
                            }
                        }
                        if (!find && num_dir < _CROSS_NUM-1) {
                            int d = num_dir;
#ifdef _DEBUG_LINES
                            line(*dbg, Point(line_end_a[l1].x * 0.3 + 263, line_end_a[l1].y * 0.3 + 168),
                                  Point(line_end_b[l1].x * 0.3 + 263, line_end_b[l1].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 2); 
#endif
                            num_direct[num_dir] = 1;
                            direction[num_dir++][0] = l1;
                        }
                    }
                }
                // find two pairs of parallel lines that has common intersetion
                if (lines > 0) {
                   int d1_list[50];
                   int d2_list[50];
                   int d1_list_size;
                   int d2_list_size;
                   for (int d1=0; d1<num_dir-1; d1++) {
                     for (int d2=d1+1; d2<num_dir; d2++) {
                       double angle = abs(v_line[direction[d1][0]][0] * v_line[direction[d2][0]][0] + v_line[direction[d1][0]][1] * v_line[direction[d2][0]][1]);
                       if (angle<0.4) {
                          if ((num_direct[d1]>1) && (d1_list_size=add_center_line(d1, exp_line_width, min_exp, max_exp, d1_list))) {
                            if ((num_direct[d2]>1) && (d2_list_size=add_center_line(d2,exp_line_width, min_exp, max_exp, d2_list))) {
                              for (int dl1=0; dl1<d1_list_size; dl1++) {
                                for (int dl2=0; dl2<d2_list_size; dl2++) {
                                  int touch=0;
                                  cen_line_size[d1] = my_abs(v_line[d1_list[2*dl1]][1] * v_line[d1_list[2*dl1+1]][2] - v_line[d1_list[2*dl1]][0] * v_line[d1_list[2*dl1+1]][3] - c[d1_list[2*dl1]]);
                                  cen_line_size[d2] = my_abs(v_line[d2_list[2*dl2]][1] * v_line[d2_list[2*dl2+1]][2] - v_line[d2_list[2*dl2]][0] * v_line[d2_list[2*dl2+1]][3] - c[d2_list[2*dl2]]);
                                  float lend11 = sqrt(sqr(line_end_a[d1_list[2*dl1]].x-line_end_b[d1_list[2*dl1]].x)+sqr(line_end_a[d1_list[2*dl1]].y-line_end_b[d1_list[2*dl1]].y));
                                  float lend12 = sqrt(sqr(line_end_a[d1_list[2*dl1+1]].x-line_end_b[d1_list[2*dl1+1]].x)+sqr(line_end_a[d1_list[2*dl1+1]].y-line_end_b[d1_list[2*dl1+1]].y));
                                  float lend21 = sqrt(sqr(line_end_a[d2_list[2*dl2]].x-line_end_b[d2_list[2*dl2]].x)+sqr(line_end_a[d2_list[2*dl2]].y-line_end_b[d2_list[2*dl2]].y));
                                  float lend22 = sqrt(sqr(line_end_a[d2_list[2*dl2+1]].x-line_end_b[d2_list[2*dl2+1]].x)+sqr(line_end_a[d2_list[2*dl2+1]].y-line_end_b[d2_list[2*dl2+1]].y));
                                  if (lend11>1.9*cen_line_size[d1] && lend21>1.9*cen_line_size[d2] && near_touch(d1_list[2*dl1], d2_list[2*dl2])) {
                                    touch++;
                                  }
                                  if (lend11>1.9*cen_line_size[d1] && lend22>1.9*cen_line_size[d2] && near_touch(d1_list[2*dl1], d2_list[2*dl2+1])) {
                                    touch++;
                                  }
                                  if (lend12>1.9*cen_line_size[d1] && lend21>1.9*cen_line_size[d2] && near_touch(d1_list[2*dl1+1], d2_list[2*dl2])) {
                                    touch++;
                                  }
                                  if (lend12>1.9*cen_line_size[d1] && lend22>1.9*cen_line_size[d2] && near_touch(d1_list[2*dl1+1], d2_list[2*dl2+1])) {
                                    touch++;
                                  }
                                  if (touch==1) {
                                    if (lend11>5*cen_line_size[d1] ||lend12>5*cen_line_size[d1] || lend21>5*cen_line_size[d2] ||lend22>5*cen_line_size[d2]) {
                                      touch=0;
                                    }
                                  }
                                  if (touch>=1 && (cen_line_size[d1]<2.3*cen_line_size[d2]) && (cen_line_size[d2]<2.3*cen_line_size[d1])) {
#ifdef _DEBUG_LINES
                                    int d = d1+d2;
                                line(*dbg, Point(line_end_a[d1_list[2*dl1]].x * 0.3 + 263, line_end_a[d1_list[2*dl1]].y * 0.3 + 168),
                                  Point(line_end_b[d1_list[2*dl1]].x * 0.3 + 263, line_end_b[d1_list[2*dl1]].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 2); 
                                line(*dbg, Point(line_end_a[d1_list[2*dl1+1]].x * 0.3 + 263, line_end_a[d1_list[2*dl1+1]].y * 0.3 + 168),
                                  Point(line_end_b[d1_list[2*dl1+1]].x * 0.3 + 263, line_end_b[d1_list[2*dl1+1]].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 2); 
                                line(*dbg, Point(line_end_a[d1_list[2*dl2]].x * 0.3 + 263, line_end_a[d1_list[2*dl1]].y * 0.3 + 168),
                                  Point(line_end_b[d1_list[2*dl2]].x * 0.3 + 263, line_end_b[d1_list[2*dl2]].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 2); 
                                line(*dbg, Point(line_end_a[d1_list[2*dl2+1]].x * 0.3 + 263, line_end_a[d1_list[2*dl2+1]].y * 0.3 + 168),
                                  Point(line_end_b[d1_list[2*dl2+1]].x * 0.3 + 263, line_end_b[d1_list[2*dl2+1]].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 2); 
#endif
                                     cen_line[d1][0] = v_line[d1_list[2*dl1]][0];
                                     cen_line[d1][1] = v_line[d1_list[2*dl1]][1];
                                     cen_line[d1][2] = (v_line[d1_list[2*dl1]][2] + v_line[d1_list[2*dl1+1]][2]) / 2.0;
                                     cen_line[d1][3] = (v_line[d1_list[2*dl1]][3] + v_line[d1_list[2*dl1+1]][3]) / 2.0;
                                     cen_num_point[d1] = line_ptr[d1_list[2*dl1]]+line_ptr[d1_list[2*dl1+1]];
                                     cen_line[d2][0] = v_line[d2_list[2*dl2]][0];
                                     cen_line[d2][1] = v_line[d2_list[2*dl2]][1];
                                     cen_line[d2][2] = (v_line[d2_list[2*dl2]][2] + v_line[d2_list[2*dl2+1]][2]) / 2.0;
                                     cen_line[d2][3] = (v_line[d2_list[2*dl2]][3] + v_line[d2_list[2*dl2+1]][3]) / 2.0;
                                     cen_num_point[d2] = line_ptr[d2_list[2*dl2]]+line_ptr[d2_list[2*dl2+1]];
                                     if (intersection_poc(d1, d2, pp)) {
                                        cross_find[cross_number] = pp;
                                        cross_num_point[cross_number] = cen_num_point[d1]+cen_num_point[d2];
                                        cross_touch[cross_number]=touch;
                                        cross_line_size[cross_number++] = (cen_line_size[d1] + cen_line_size[d2])/2.0;
                                     }
                                  }
                                }
                              }
                            } else {
                              for (int dd2=0; dd2<num_direct[d2];dd2++) {
                                float len = sqrt(sqr(line_end_a[direction[d2][dd2]].x-line_end_b[direction[d2][dd2]].x)+sqr(line_end_a[direction[d2][dd2]].y-line_end_b[direction[d2][dd2]].y));
                                if (len>3.5*cen_line_size[d1] && near_touch(direction[d1][cen_min_ind[d1]],direction[d2][dd2]) && near_touch(direction[d1][cen_min_ind[d1]+1],direction[d2][dd2])) {
                                  cen_line[d2][0] = v_line[direction[d2][dd2]][0];
                                  cen_line[d2][1] = v_line[direction[d2][dd2]][1];
                                  cen_line[d2][2] = v_line[direction[d2][dd2]][2];
                                  cen_line[d2][3] = v_line[direction[d2][dd2]][3];
                                  cen_line_size[d2] = cen_line_size[d1];
                                  cen_num_point[d2] = line_ptr[direction[d2][dd2]];
                                  cen_min_ind[d2]=dd2;
                                  if (intersection_poc(d1, d2, pp)) {
                                    cross_find[cross_number] = pp;
                                    cross_touch[cross_number]=2;
                                    cross_num_point[cross_number] = cen_num_point[d1] + cen_num_point[d2];
                                    cross_line_size[cross_number++] = (cen_line_size[d1] + cen_line_size[d2])/2.0;
#ifdef _DEBUG_LINES
                                    int d = d1+dd2;
                                line(*dbg, Point(line_end_a[direction[d2][dd2]].x * 0.3 + 263, line_end_a[direction[d2][dd2]].y * 0.3 + 168),
                                  Point(line_end_b[direction[d2][dd2]].x * 0.3 + 263, line_end_b[direction[d2][dd2]].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 3); 
                                line(*dbg, Point(line_end_a[direction[d1][cen_min_ind[d1]]].x * 0.3 + 263, line_end_a[direction[d1][cen_min_ind[d1]]].y * 0.3 + 168),
                                  Point(line_end_b[direction[d1][cen_min_ind[d1]]].x * 0.3 + 263, line_end_b[direction[d1][cen_min_ind[d1]]].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 2); 
                                line(*dbg, Point(line_end_a[direction[d1][cen_min_ind[d1]+1]].x * 0.3 + 263, line_end_a[direction[d1][cen_min_ind[d1]+1]].y * 0.3 + 168),
                                  Point(line_end_b[direction[d1][cen_min_ind[d1]+1]].x * 0.3 + 263, line_end_b[direction[d1][cen_min_ind[d1]+1]].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 2); 
#endif
                                  }
                                }
                              }
                            }
                          } else {
                            if ((num_direct[d2]>1) && (d2_list_size=add_center_line(d2,exp_line_width, min_exp, max_exp, d2_list))) {
                              for (int dd1=0; dd1<num_direct[d1];dd1++) {
                                float len = sqrt(sqr(line_end_a[direction[d1][dd1]].x-line_end_b[direction[d1][dd1]].x)+sqr(line_end_a[direction[d1][dd1]].y-line_end_b[direction[d1][dd1]].y));
                                if (len>3.5*cen_line_size[d2] && near_touch(direction[d2][cen_min_ind[d2]],direction[d1][dd1]) && near_touch(direction[d2][cen_min_ind[d2]+1],direction[d1][dd1])) {
                                  cen_line[d1][0] = v_line[direction[d1][dd1]][0];
                                  cen_line[d1][1] = v_line[direction[d1][dd1]][1];
                                  cen_line[d1][2] = v_line[direction[d1][dd1]][2];
                                  cen_line[d1][3] = v_line[direction[d1][dd1]][3];
                                  cen_line_size[d1] = cen_line_size[d2];
                                  cen_num_point[d1] = line_ptr[direction[d1][dd1]];
                                  cen_min_ind[d1]=dd1;
                                  if (intersection_poc(d1, d2, pp)) {
                                    cross_find[cross_number] = pp;
                                    cross_touch[cross_number]=2;
                                    cross_num_point[cross_number] = cen_num_point[d1] + cen_num_point[d2];
                                    cross_line_size[cross_number++] = (cen_line_size[d1] + cen_line_size[d2])/2.0;
#ifdef _DEBUG_LINES
                                    int d = dd1+d2;
                                line(*dbg, Point(line_end_a[direction[d1][dd1]].x * 0.3 + 263, line_end_a[direction[d1][dd1]].y * 0.3 + 168),
                                  Point(line_end_b[direction[d1][dd1]].x * 0.3 + 263, line_end_b[direction[d1][dd1]].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 3); 
                                line(*dbg, Point(line_end_a[direction[d2][cen_min_ind[d2]]].x * 0.3 + 263, line_end_a[direction[d2][cen_min_ind[d2]]].y * 0.3 + 168),
                                  Point(line_end_b[direction[d2][cen_min_ind[d2]]].x * 0.3 + 263, line_end_b[direction[d2][cen_min_ind[d2]]].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 2); 
                                line(*dbg, Point(line_end_a[direction[d2][cen_min_ind[d2]+1]].x * 0.3 + 263, line_end_a[direction[d2][cen_min_ind[d2]+1]].y * 0.3 + 168),
                                  Point(line_end_b[direction[d2][cen_min_ind[d2]+1]].x * 0.3 + 263, line_end_b[direction[d2][cen_min_ind[d2]+1]].y * 0.3 + 168), Scalar(bcolors[d%7][0], bcolors[d%7][1], bcolors[d%7][2]), 2); 
#endif
                                  }
                                }
                              }
                            }
                          }
                       }
                     }
                   }
                    poc_center = 0;
                    for (int l1 = 0; l1 < lines; l1++) {
                        lin[l1].clear();
                        line_ptr[l1] = 0;
                    }
                    lines = 0;
                }
                x++;
                ptr++;
            }
        }
        

        // cross number represents number of two pairs of parallel lines that has common intersection
        for (int ii = 0; ii < cross_number; ii++) {
            cross_circles[ii] = 0;
            for (int cc = 0; cc < num_circles; cc++) {
                if (dist(cross_find[ii], circ_center[cc]) < circ_radius[cc] * 0.6) {
                    cross_circles[ii]++;
                }
            }
        }
        int cross_ind = 0;
        for (int ii = 1; ii < cross_number; ii++) {
            if (cross_circles[ii] > cross_circles[cross_ind]
                    || (cross_circles[ii] == cross_circles[cross_ind] && cross_num_point[ii] > cross_num_point[cross_ind])) {
                cross_ind = ii;
            }
        }

        cout << "CROSS num "<<cross_number<<" line_size "<<cross_line_size[0]<<" num point "<<cross_num_point[0]<<endl;
        int ct4=0;
        int ct4_index=-1;
        for (int cc=0; cc<cross_number; cc++) {
          if (cross_touch[cc]>=3) {
            ct4++;
            ct4_index=cc;
          }
        }
        if (cross_number > 0 && ct4==1) {
        // there is only 1 candidate with intersection of 4 lines _| |_
        //                                                        _   _
        //                                                         | |
            last_line_size = cross_line_size[ct4_index];
            last_ell_size = last_line_size *10;
            ell.center = cross_find[ct4_index];
            ret = true;
            cout << "FIND cross CT4 " << ell.center << " max sum " << cross_num_point[cross_ind] << endl;
            
        } else if (cross_number > 0 && (cross_circles[cross_ind] > 0 || (cross_number==1 &&
        ((cross_line_size[0]>100 && cross_num_point[0]>400)||
        (cross_line_size[0]>200 && cross_num_point[0]>200))))) {
        // there are candidates with intersection of 3 lines _| |_
            last_line_size = cross_line_size[cross_ind];
            last_ell_size = last_line_size *10;
            ell.center = cross_find[cross_ind];
            ret = true;
            cout << "FIND cross " << ell.center << " max sum " << cross_num_point[cross_ind] << endl;
        } else {
        // there was detected parts of circle that satisfy the criteria for arc length
            float x_avg = 0, y_avg = 0, rad_avg = 0.0;
            int circ_ind = -1;
            int max_circ = -1;
            cout << "Circles "<<num_circles<<endl;
            for (int cc = 0; cc < num_circles; cc++) {
                circ_to_circ[cc] = 1;
                for (int cc2 = cc + 1; cc2 < num_circles; cc2++) {
                    if (dist(circ_center[cc], circ_center[cc2]) < circ_radius[cc] * 0.6) {
                        circ_to_circ[cc]++;
                        circ_num_point[cc] += circ_num_point[cc2];
                    }
                }
                if (circ_to_circ[cc] > max_circ) {
                    max_circ = circ_to_circ[cc];
                    circ_ind = cc;
                }
            }
            // these arcs have together more than 700 pixels
            if (circ_ind>=0 && circ_num_point[circ_ind] > 700) {
                for (int cc = 0; cc < num_circles; cc++) {
                    if (dist(circ_center[cc], circ_center[circ_ind]) < circ_radius[circ_ind] * 0.6) {
                        x_avg += circ_center[cc].x;
                        y_avg += circ_center[cc].y;
                        rad_avg += circ_radius[cc];
                    }
                }
                ell.center.x = x_avg / circ_to_circ[circ_ind];
                ell.center.y = y_avg / circ_to_circ[circ_ind];
                last_ell_size = (rad_avg / circ_to_circ[circ_ind])/0.9;
                ret = true;
                cout << "FIND cross circle " << ell.center << " max sum " << circ_num_point[circ_ind] << " ell size " << last_ell_size << endl;
            }
        }

#ifdef _DEBUG_LINES
        cout << "Cross "<< ell.center.x<<","<<ell.center.y<<endl;
        line(*dbg, Point((ell.center.x -50)* 0.3 + 263, ell.center.y * 0.3 + 168),
                Point((ell.center.x+50) * 0.3 + 263, ell.center.y * 0.3 + 168), Scalar(0,0,255), 1); 
        line(*dbg, Point(ell.center.x* 0.3 + 263, (ell.center.y-50) * 0.3 + 168),
                Point(ell.center.x * 0.3 + 263, (ell.center.y+50) * 0.3 + 168), Scalar(0,0,255), 1); 
        imshow("test", *dbg);
#endif
    }
    return ret;
}

static int hist[256];
// find big cross and part of the circle if other methods were unsuccesful
static bool findBigTarget(OutputArray _pic, InputArray _orig, RotatedRect &ell) {
    bool ret = false;
    Mat orig = _orig.getMat();
    Size size = orig.size();
    long int sum = 0, poc = 0;
    int min = 255, max = 0;
    int min_sat = 0, max_sat = 0;
    int border_x = 10, border_y = 10;
    int ii;
    int lok_min_min, lok_min_max, min_size;

    for (ii = 0; ii < 256; ii++) {
        hist[ii] = 0;
    }
    // compute values for setting exposure
    if (orig.isContinuous()) {
        int x, y, next_line = size.width;
        uchar *o = orig.ptr(0);
        int p_ptr = 0;

        for (y = 0; y < size.height; y += 4) {
            for (x = 0; x < size.width; x++) {
                hist[o[p_ptr]]++;
                sum += o[p_ptr++];
            }
            p_ptr += 3 * next_line;
            poc += next_line;
        }
        min = -1;
        lok_min_max = -1;
        lok_min_min = -1;
        min_size = -1;
        for (ii = 0; ii < 256; ii++) {
            if (min < 0 && hist[ii] > 0) {
                min = ii;
            }
            if (min_size < 0 && hist[ii] > 150) {
                min_size = ii;
            }
            if ((min_size > 0) && (hist[ii] < 100)) {
                min_size = ii;
                break;
            }
        }
        for (ii = min_size; ii > min; ii--) {
            if (lok_min_max < 0 && (hist[ii] + hist[ii - 1]) < (hist[ii + 1] + hist[ii])) {
                lok_min_max = ii;
            }
            if (lok_min_max > 0 && (hist[ii] + hist[ii - 1]) > (hist[ii + 1] + hist[ii])) {
                lok_min_min = ii;
                break;
            }
        }
        for (ii = 255; ii > 0; ii--) {
            if (hist[ii] > 0) {
                max = ii;
                break;
            }
        }
        last_col_min = min;
        last_col_max = max;
        last_bl_sat = hist[0];
        last_wh_sat = hist[255];
        // find cross and part of the circle
        ret = findBigCross(_pic, _orig, ell);
    }
    return ret;
}

// detect minimal and maximal intensity values for image to set correct exposure
static bool exposure_detect(InputArray _orig) {
    bool ret = false;
    Mat orig = _orig.getMat();
    Size size = orig.size();
    long int sum = 0, poc = 0;
    int min = 255, max = 0;
    int min_sat = 0, max_sat = 0;
    int ii;
    int lok_min_min, lok_min_max, min_size;

    for (ii = 0; ii < 256; ii++) {
        hist[ii] = 0;
    }

    if (orig.isContinuous()) {
        int x, y, next_line = size.width;
        uchar *o = orig.ptr(0);
        int p_ptr = 0;

        for (y = 0; y < size.height; y += 4) {
            for (x = 0; x < size.width; x++) {
                hist[o[p_ptr]]++;
                sum += o[p_ptr++];
            }
            p_ptr += 3 * next_line;
            poc += next_line;
        }
        min = -1;
        lok_min_max = -1;
        lok_min_min = -1;
        min_size = -1;
        for (ii = 0; ii < 256; ii++) {
            if (min < 0 && hist[ii] > 0) {
                min = ii;
            }
            if (min_size < 0 && hist[ii] > 150) {
                min_size = ii;
            }
            if ((min_size > 0) && (hist[ii] < 100)) {
                min_size = ii;
                break;
            }
        }
        for (ii = min_size; ii > min; ii--) {
            if (lok_min_max < 0 && (hist[ii] + hist[ii - 1]) < (hist[ii + 1] + hist[ii])) {
                lok_min_max = ii;
            }
            if (lok_min_max > 0 && (hist[ii] + hist[ii - 1]) > (hist[ii + 1] + hist[ii])) {
                lok_min_min = ii;
                break;
            }
        }
        for (ii = 255; ii > 0; ii--) {
            if (hist[ii] > 0) {
                max = ii;
                break;
            }
        }
        last_col_min = min;
        last_col_max = max;
        last_bl_sat = hist[0];
        last_wh_sat = hist[255];
    }
}


// initialization of global structures
void detect_init() {
    element = getStructuringElement(MORPH_RECT, Size(2 * box_size + 1, 2 * box_size + 1), Point(box_size, box_size));
    erode_el = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    element_er_cross = getStructuringElement(MORPH_CROSS, Size(5, 5));
    element_er = getStructuringElement(MORPH_RECT, Size(3, 3));
    last_detect = 100;
    cross_det = 0;
    skip=0;
    
    for (int ii=0; ii<MAX_LINES; ii++) {
          lin[ii].reserve(800);
    }
}

static void drawEllipse(Mat& img, const RotatedRect& box, const Scalar& color, int thickness = 1, int lineType = 8) {
    RotatedRect new_box = box;
    new_box.size.height = box.size.height - 0.1 * box.size.height;
    new_box.size.width = box.size.width - 0.1 * box.size.width;
    
    ellipse(img, new_box, color, 1);

    new_box = box;
    new_box.size.height = box.size.height + 0.1 * box.size.height;
    new_box.size.width = box.size.width + 0.1 * box.size.width;

    ellipse(img, new_box, color, thickness);
}

static int img_index = 0;

#define _MAX_LIMIT 900
#define _MIN_LIMIT 60

int detect_cr(Mat src, Mat *img, exposure_balance *eb, RotatedRect &ell, bool use_gui, float &distance) {
    bool found = false, found_big = false;
    Mat rvec(3, 1, DataType<double>::type), tvec(3, 1, DataType<double>::type);

    glob_dist = distance;
    distance=0.0;
    Size pic_size = src.size();
    height = pic_size.height;
    width = pic_size.width;
    vector<Vec3f> circles;
    double t = (double) getTickCount();
    if (img == NULL && use_gui) {
        cvtColor(src, tmp_img, CV_GRAY2RGB);
        img = &tmp_img;
    }

#ifdef _DEBUG_IMG  
    orig = src;
#endif
    dbg = img;

    if (glob_dist < 9.5 && glob_dist>0.4) {
        // perform adaptive thresholding see Section 2.1.
        adaptiveThreshold(src, tmp, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 2 * box_size + 1, 1);
#ifdef _DEBUG_OUTPUT
        imshow("Adaptive middle", tmp);
#endif
        found = findTarget(tmp, src, ell, false);

        if (found) {
            double ratio;
            if (ell.size.height < ell.size.width) {
                ratio = ell.size.height / ell.size.width;
            } else {
                ratio = ell.size.width / ell.size.height;
            }
            float x1 = (((ell.center.x - cam_cx) - cam_dx) / (1 + (cam_dx * (ell.center.x-cam_cx)) / (cam_fx * cam_fx))) / cam_fx;
            float y1 = (((ell.center.y - cam_cy) - cam_dy) / (1 + (cam_dy * (ell.center.y-cam_cy)) / (cam_fy * cam_fy))) / cam_fy;
            float d2 = (x1 * x1 + y1 * y1);
            float koef = sqrt(1 + d2);
            // test if the detected ellipse size is suitable for UAV's altitude
            // The limits MIN_LIMT and MAX LIMIT are not precise because our altitude measurement was not good enough
            found = ((last_ell_size * glob_dist) > _MIN_LIMIT) && ((last_ell_size * glob_dist) < _MAX_LIMIT);
            distance = koef * cam_scale / last_ell_size;
        }
    }
    if (found) {
        if (use_gui) {
            drawEllipse(*img, ell, Scalar(255, 255, 0), 2);
        }
        cross_det = 1;
    } else {
        // if first attempt didn't find target than try big or small tedection with respect to UAV's altitude
        if (glob_dist >= 2.5) {
            // perform adaptive thresholding with box size 5 pixel see Section 2.1.
            adaptiveThreshold(src, tmp3, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 2 * small_box_size + 1, 1);
#ifdef _DEBUG_OUTPUT
            imshow("Adaptive-small", tmp3);
#endif
            found = findTarget(tmp3, src, ell, true);
            if (found) {
                double ratio;
                if (ell.size.height < ell.size.width) {
                    ratio = ell.size.height / ell.size.width;
                } else {
                    ratio = ell.size.width / ell.size.height;
                }
                float x1 = (((ell.center.x - cam_cx) - cam_dx) / (1 + (cam_dx * (ell.center.x-cam_cx)) / (cam_fx * cam_fx))) / cam_fx;
                float y1 = (((ell.center.y - cam_cy) - cam_dy) / (1 + (cam_dy * (ell.center.y-cam_cy)) / (cam_fy * cam_fy))) / cam_fy;
                float d2 = (x1 * x1 + y1 * y1);
                float koef = sqrt(1 + d2);
                // test if the detected ellipse size is suitable for UAV's altitude
                // The limits MIN_LIMT and MAX LIMIT are not precise because our altitude measurement was not good enough
                found = ((last_ell_size * glob_dist) > _MIN_LIMIT) && ((last_ell_size * glob_dist) < _MAX_LIMIT);
                distance = koef * cam_scale / last_ell_size;
                if (found) {
                    cross_det = 1;
                    if (use_gui) {
                        drawEllipse(*img, ell, Scalar(255, 255, 0), 2);
                    }
                }
            }
        }
        if (!found && (glob_dist < 2.0)) {
            // the UAV is very near to target, try to find big cross and part of circle
            found_big = findBigTarget(tmp, src, ell);
            if (found_big) {
                float x1 = (ell.center.x - cam_cx)- cam_dx;
                float y1 = (ell.center.y - cam_cy)- cam_dy;
                float d2 = (x1 * x1 + y1 * y1);
                float koef = sqrt(cam_fx * cam_fx + d2) / cam_fx;
                // test if the detected ellipse size is suitable for UAV's altitude
                // The limits MIN_LIMT and MAX LIMIT are not precise because our altitude measurement was not good enough
                found_big = ((last_ell_size * glob_dist) > _MIN_LIMIT) && ((last_ell_size * glob_dist) < _MAX_LIMIT);
                distance = (cam_scale   / last_ell_size);
                if ((distance>glob_dist+2) || (distance<glob_dist-1.5)) {
                    distance = 0.0;
                }
                if (found_big && use_gui) {
                    circle(*img, Point(cvRound(ell.center.x), cvRound(ell.center.y)), last_line_size / 4, Scalar(255, 255, 0), -1, CV_AA);
                }
            }
        } else if (!found) {
            // if target was not detected compute new parameters for exposure
            exposure_detect(src);
        }
    }

#ifdef _DEBUG_IMG
    imshow("src", *img);
    int key = waitKey(1) & 0xff;
    if (key == 's' && !found && !found_big) {
        char name[100];
        snprintf(name, 90, "/home/petr/img%05i.bmp", img_index++);
        imwrite(name, src);
        cout << "Saved image " << name << endl;
    }
#endif

    if (!found && !found_big && last_detect < 100) {
        last_detect++;
        last_detect = 100;
    }
    eb->change_exp = 0;
    if (found) {
        // if target was founded than use information about target white and black to change exposure
        if ((last_col_max > 250 || last_col_min > 120) && 0.5 * eb->exposure_value > 15) {
            if (0.5 * eb->exposure_value > 15) {
               eb->change_exp = 0.5 * eb->exposure_value;
            } else {
               eb->change_exp = 15;
            }
        } else if ((last_col_max > 220 || last_col_min > 40)) {
            if (0.85 * eb->exposure_value > 15) {
               eb->change_exp = 0.85 * eb->exposure_value;
            } else {
               eb->change_exp = 15;
            }
        } else if ((last_col_min < 40 || last_col_max < 200) && 1.1 * eb->exposure_value < 10000) {
            eb->change_exp = 1.15 * eb->exposure_value;
        } else if ((last_col_min < 20 || last_col_max < 130) && 1.4 * eb->exposure_value < 10000) {
            eb->change_exp = 1.4 * eb->exposure_value;
        }
    } else {
        // if no target was founded than use information about whole image to change exposure
        if (last_col_max > 240 && 0.85 * eb->exposure_value > 15) {
            if (last_wh_sat > 10000) {
                eb->change_exp = (0.5 * eb->exposure_value < 15) ? 15 : 0.5 * eb->exposure_value;
            } else {
                eb->change_exp = (0.85 * eb->exposure_value < 15) ? 15 : 0.85 * eb->exposure_value;
            }
        } else if (last_col_min < 20 && 1.15 * eb->exposure_value < 10000) {
            if (last_bl_sat > 10000) {
                eb->change_exp = (1.4 * eb->exposure_value > 10000) ? 10000 : 1.4 * eb->exposure_value;
            } else {
                eb->change_exp = 1.15 * eb->exposure_value;
            }
        }
    }
    eb->wh_satur = last_wh_sat;
    eb->bl_satur = last_bl_sat;
    eb->wh_lvl = last_col_max;
    eb->bl_lvl = last_col_min;

    if (found || found_big) {
        // update position with respect to camera tilt - cam_dx and cam_dy parameters
        last_detect=0;
        last_ell.center.x=ell.center.x;
        last_ell.center.y=ell.center.y;
        float dist = distance;
        if (dist<0.01) {
          dist = glob_dist;
        }
        float x1 = ell.center.x - cam_cx;
        float rx = dist*((x1 - cam_dx) / (1 + (cam_dx * x1) / (cam_fx * cam_fx))) / cam_fx;
        float y1 = ell.center.y - cam_cy;
        float ry = dist*((y1 - cam_dy) / (1 + (cam_dy * y1) / (cam_fy * cam_fy))) / cam_fy;
    }

    return (found || found_big) ? 1 : 0;
}
