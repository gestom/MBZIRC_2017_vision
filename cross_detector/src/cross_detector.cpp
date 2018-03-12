/*
*  cross detector.cpp
*  
*  ROS node - landing pattern detection for MBZIRC competition
*  
*  Created on: 7. 1. 2016
*  Author: petr stepan
*/


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <geometry_msgs/PoseStamped.h>
#include "cross_detector/detector.h"
#include "cross_detector/cross_alt.h"

#include "tf/transform_listener.h"
#include "tf/message_filter.h"
#include "message_filters/subscriber.h"
#include <std_msgs/Int32.h>
#include <std_msgs/Float32.h>
#include <dynamic_reconfigure/server.h>
#include <cross_detector/cross_detector_drsConfig.h>

static const std::string OPENCV_WINDOW = "Cross Detector";
namespace enc = sensor_msgs::image_encodings;


// Camera shift to the center of the UAV
const float position_of_camera [] = {0.1325, 0, 0};

/**
 * @brief The CrossDetector class
 */
class CrossDetector {
private:
    bool gui_;
    float dd_x, dd_y;
    
    float cross_expected_altitude_;
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    tf::TransformListener* tf_listener_;
    std::string target_frame_;   // local origin
    std::string frame_name_;     // fcu_uavXX
    ros::Publisher cross_pub_;
    ros::Publisher exposure_time_pub_;
    ros::Publisher altitude_pub_;
    image_transport::CameraSubscriber cam_sub_;
    geometry_msgs::PoseStamped pose, drone_pose;
    
    
    int first_image;
    int img_index;
    cv::Mat intrinsic;
    cv::Mat distCoeffs;
    float delta_x, delta_y;
    float cam_fx, cam_fy;
    float cam_cx, cam_cy;
    float delay;
    float orig_x, orig_y, orig_z;
    float cam_dx;
    float cam_dy;
    float alt;
    float max_dif;
    int time_delay[15];
    
    int last_pose;
    float last_pose_x, last_pose_y;
    
    bool transform_cross_;
    
    /// parameters for expose change
    exposure_balance ex_balance_;
    int exposure_cnt_;
    int exposure_avg_;
    bool exposure_changed_;
    ros::Time exposure_last_change_;
    ros::Duration exposure_duration_;
    
    int fps_cnt;
    int det_cnt;
    int ptr_shift;
    int stat_see[3], frame_cnt[3];
    int stat_cnt;
    int last_id;
    ros::Time fps_time_old;
    
public:
    // constructor
    CrossDetector();
    // destructor
    ~CrossDetector();
    // callback for image processing
    void cameraCallback(const sensor_msgs::ImageConstPtr& image_msg,
                        const sensor_msgs::CameraInfoConstPtr& info_msg);
private:    
    // compute position of the target from detected image and position of the UAV
    bool cameraPosition(float loc_delay, float cx, float cy, float &altitude, const ros::Time stamp);
};


/**
 * @brief CrossDetector constructor
 */
CrossDetector::CrossDetector():it_(nh_)
{
    ros::NodeHandle private_node_handle("~");
    private_node_handle.param("target_frame", target_frame_, std::string("local_origin"));
    private_node_handle.param("gui", gui_, bool(false));
    private_node_handle.param("frame_topic", frame_name_, std::string(""));
    private_node_handle.param("pattern_expected_altitude", cross_expected_altitude_, float(0));
    private_node_handle.param("transform_cross", transform_cross_, bool(true));
    
    ROS_INFO("Starting CROSS_DETECTOR node with gui_setting: %s", gui_? "TRUE": "FALSE");
    
    delta_x = delta_y = 0.0;
    std::string camera_calib_file;
    private_node_handle.param("camera_calib_file", camera_calib_file, std::string(""));
    ROS_INFO_STREAM("using camera calibration file: "<<camera_calib_file);
    FILE *f=fopen(camera_calib_file.c_str(),"r");
    if (f!=NULL) {
        if (fscanf(f, "%f %f", &delta_x, &delta_y)!=2) {
            delta_x = delta_y = 0.0;
        }
        fclose(f);
    }
    
    max_dif=0.0;
    // publisher for detected position of the cross
    cross_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("cross_position", 1);
    exposure_time_pub_ = nh_.advertise<std_msgs::Int32>("exposure_time", 1);
    altitude_pub_ = private_node_handle.advertise<cross_detector::cross_alt>("altitude", 1);
    img_index=0;
    
    // subscriber for image and camera_info from camera
    cam_sub_ = it_.subscribeCamera("camera", 1, &CrossDetector::cameraCallback, this);
    
    tf_listener_ = new tf::TransformListener();
    exposure_changed_ = false;
    ex_balance_.exposure_value = 1000;
    if (gui_) {
        cv::namedWindow(OPENCV_WINDOW);
    }
    std_msgs::Int32 msg;
    msg.data = ex_balance_.exposure_value;
    exposure_time_pub_.publish(msg);
    exposure_cnt_ = 0;
    fps_cnt = 1000;
    det_cnt = 0;
    first_image=1;
    fps_time_old = ros::Time::now();
    detect_init();
    delay=-0.03;
    alt=0.0;
    stat_cnt=0;
    stat_see[0]=stat_see[1]=stat_see[2]=0;
    frame_cnt[0]=frame_cnt[1]=frame_cnt[2]=0;
    last_id = 2;
}


/**
 * @brief CrossDetector destructor
 */
CrossDetector::~CrossDetector()
{
    if (gui_) cv::destroyWindow(OPENCV_WINDOW);
}

#define sqr(a) ((a)*(a))
#define my_abs(a) ((a<0)?(-(a)):(a))

// compute target position from camera position, and from size of detected ellipse
// see Section 2.5
bool CrossDetector::cameraPosition(float loc_delay, float cx, float cy, float &altitude, const ros::Time stamp) {
    bool ret = 0;
    int index=0;
    
    geometry_msgs::PoseStamped pose_of_camera, tf_pose_of_camera, pose_of_camera2, tf_pose_of_camera2 , tk_pose, tf_tk_pose;
    pose_of_camera.header.frame_id = frame_name_;
    if (loc_delay<0) {
        pose_of_camera.header.stamp = stamp - ros::Duration(-loc_delay);
    } else {
        pose_of_camera.header.stamp = stamp + ros::Duration(loc_delay);
    }
    pose_of_camera.header.stamp = stamp- ros::Duration(0.4);
    pose_of_camera.pose.position.x = position_of_camera[0];
    pose_of_camera.pose.position.y = position_of_camera[1];
    pose_of_camera.pose.position.z = position_of_camera[2];
    pose_of_camera.pose.orientation.x = 0.0;
    pose_of_camera.pose.orientation.y = 0.0;
    pose_of_camera.pose.orientation.z = 0.0;
    pose_of_camera.pose.orientation.w = 1.0;
    
    // vector that describe detected position of cross in fcu frame
    geometry_msgs::Vector3Stamped cross_cam_pose, tf_cross_cam_pose;
    cross_cam_pose.header = pose_of_camera.header;
    float x1 = cx - cam_cx;
    float y1 = cy - cam_cy;
    float alpha=-M_PI/2.0;
    
    x1 = ((x1 - cam_dx) / (1 + (cam_dx * x1) / (cam_fx * cam_fx))) / cam_fx;
    y1 = ((y1 - cam_dy) / (1 + (cam_dy * y1) / (cam_fy * cam_fy))) / cam_fy;
    
    cross_cam_pose.vector.x = cos(alpha)*x1-sin(alpha)*y1;
    cross_cam_pose.vector.y = -sin(alpha)*x1-cos(alpha)*y1;
    cross_cam_pose.vector.z = - 1;
    
    try {
        // compute u camer position for equation 3 from section 2.5
        tf_listener_->transformPose(target_frame_, pose_of_camera, tf_pose_of_camera);

        // compute R*x from equation 3 from section 2.5
        tf_listener_->transformVector(target_frame_, cross_cam_pose, tf_cross_cam_pose);
    }
    catch (tf::TransformException &ex) {
        ROS_ERROR("TF exception:\n%s",ex.what());
        time_delay[14]++;
        return ret;
    }
    double vec_size = sqrt(tf_cross_cam_pose.vector.x*tf_cross_cam_pose.vector.x+tf_cross_cam_pose.vector.y*tf_cross_cam_pose.vector.y+tf_cross_cam_pose.vector.z*tf_cross_cam_pose.vector.z);
    double tt = altitude / vec_size;
    if (my_abs(tf_cross_cam_pose.vector.z)<0.00001) {
        ROS_ERROR("Z vector small %f\n",tf_cross_cam_pose.vector.z);
        return ret;
    }
    
    altitude = -tf_cross_cam_pose.vector.z*tt;
    
    // equation 3 from section 2.5
    pose.pose.position.x = tf_pose_of_camera.pose.position.x + tt * tf_cross_cam_pose.vector.x;
    pose.pose.position.y = tf_pose_of_camera.pose.position.y + tt * tf_cross_cam_pose.vector.y;
    pose.pose.position.z = tf_pose_of_camera.pose.position.z + tt * tf_cross_cam_pose.vector.z;
    
    ret = 1;
    return ret;
}


/**
 * @brief CameraCallback - method for processing an image
 * @param msg - image which should contains the cross
 */
void CrossDetector::cameraCallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg)
{
    cv_bridge::CvImagePtr image;
    cv::Mat src, image_debug;
    cv::RotatedRect cross_pose;
    float saved_altitude=-1000.0;
    float altitude;
    geometry_msgs::PoseStamped best_pose;
    int key;
    
    // try to find transform from fcu frame to target frame
    ros::Time acquisition_time = info_msg->header.stamp;
    ROS_INFO("Image from time %i.%i delay %f\n", acquisition_time.sec, acquisition_time.nsec, delay);
    
    try {
        ros::Duration timeout(0.1); // sec
        tf_listener_->waitForTransform(target_frame_, frame_name_, acquisition_time, timeout);
    }
    catch (tf::TransformException& ex) {
        ROS_WARN("TF exception:\n%s", ex.what());
        return;
    }
    
    tf::StampedTransform transform;
    try {
        tf_listener_->lookupTransform(target_frame_, frame_name_, acquisition_time, transform);
        pose.header.frame_id = frame_name_;
        pose.header.stamp = info_msg->header.stamp - ros::Duration(0.4);
        pose.pose.position.x=0;
        pose.pose.position.y=0;
        pose.pose.position.z=0;
        pose.pose.orientation.x=0;
        pose.pose.orientation.y=0;
        pose.pose.orientation.z=0;
        pose.pose.orientation.w=1;
        tf_listener_->transformPose("local_origin", pose, drone_pose);
    } catch (tf::TransformException &ex) {
        ROS_ERROR("XXXX %s",ex.what());
    }
    
    alt = saved_altitude = altitude = transform.getOrigin().z()-cross_expected_altitude_-0.18; // camera shift 0.18
    ROS_INFO("Altitude is %f fcu x,y(%f, %f) z coord %f\n", altitude,transform.getOrigin().x(), transform.getOrigin().y(), transform.getOrigin().z());

    bool is_detected;
    /// Try to find cross in camera image
    try {
        if (first_image==1) {
            intrinsic = (cv::Mat_<float>(3,3) << info_msg->K[0], info_msg->K[1], info_msg->K[2], info_msg->K[3], info_msg->K[4], info_msg->K[5], info_msg->K[6], info_msg->K[7], info_msg->K[8]);
            cam_fx = info_msg->K[0];
            cam_fy = info_msg->K[4];
            cam_cx = info_msg->K[2];
            cam_cy = info_msg->K[5];
            distCoeffs = (cv::Mat_<float>(1,5) << -0.0448, 0.0686053, -0.072277, 0.027011955, 0.0);
            set_camera_param(intrinsic, distCoeffs, delta_x, delta_y);
            first_image = 0;
        } 
        if (enc::isColor(image_msg->encoding)) {
            // transfer msg image to opencv image
            image = cv_bridge::toCvCopy(image_msg, enc::BGR8);
            // transfer colored image to grayscale image
            cv::cvtColor(image->image, src, CV_RGB2GRAY);
            // find cross
            is_detected = detect_cr(src, &image->image, &ex_balance_, cross_pose, gui_, altitude);
        } else {
            // transfer msg image to opencv image
            image = cv_bridge::toCvCopy(image_msg, enc::MONO8);
            cvtColor(image->image, image_debug, CV_GRAY2RGB);
            // find cross
            is_detected = detect_cr(image->image, &image_debug, &ex_balance_, cross_pose, gui_, altitude);
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Failed to convert image: %s", e.what());
        return;
    }

    /// EXPOSURE TIME FOR CAMERA
    if (ex_balance_.change_exp!=0) {
        if (exposure_cnt_==0) {
            exposure_avg_ = ex_balance_.change_exp;
        } else {
            exposure_avg_ = (int)((exposure_avg_*(double)exposure_cnt_+ex_balance_.change_exp)/((double)exposure_cnt_+1.0));
        }
        exposure_cnt_++;
        if (exposure_changed_) {
            exposure_duration_ = image_msg->header.stamp - exposure_last_change_;
        }
        if (exposure_cnt_>5 && (!exposure_changed_ || (exposure_duration_.sec>1) || (exposure_duration_.nsec>150000000L))) {
            std_msgs::Int32 msg_exposure_time;
            ex_balance_.exposure_value = (ex_balance_.exposure_value+exposure_avg_)/2;
            exposure_avg_ = 0;
            exposure_cnt_ = 0;
            exposure_changed_ = true;
            exposure_last_change_ = image_msg->header.stamp;
            msg_exposure_time.data = ex_balance_.exposure_value; 
            exposure_time_pub_.publish(msg_exposure_time);
        }
    }
    /// Update GUI Window - for debugging
    if (gui_){
        if (enc::isColor(image_msg->encoding)) {
            cv::imshow(OPENCV_WINDOW, image->image);
        }else{
            cv::imshow(OPENCV_WINDOW, image_debug);
        }
        key = cv::waitKey(1) & 0xff;
        if (key == 's') {
            char name[100];
            snprintf(name, 90, "/home/mrs/calib/img%05i.bmp", img_index++);
            cv::imwrite(name, src);
            std::cout << "Saved image " << name << std::endl;
        }
    } 
    
    orig_x = -2.87;
    orig_y = 0.58;
    
    if (is_detected){
        det_cnt++;
    }
    // If a pattern is found then publish its position
    if (transform_cross_ && is_detected) {
        bool unknown_alt = false;
        cross_detector::cross_alt msg;
        // use timestamp from the image message
        msg.header.stamp = image_msg->header.stamp;
        if (altitude<=0.0) {
            altitude = (saved_altitude<0.12) ? 0.12 : saved_altitude;
            unknown_alt = true;
        }
        cam_dx = delta_x;
        cam_dy = delta_y;
        
        // the position of target from the position of camera
        if (cameraPosition(delay, cross_pose.center.x, cross_pose.center.y, altitude, image_msg->header.stamp)) {
            // publish computed altitude with buffer
            if (last_pose<5) {
                float dif1 =  sqrt(sqr(pose.pose.position.x-last_pose_x)+sqr(pose.pose.position.y-last_pose_y));
                if (dif1<1.0) {
                    max_dif=dif1;
                    pose.header = image_msg->header;
                    pose.header.frame_id="local_origin";
                    pose.pose.orientation.x = 0.0;
                    pose.pose.orientation.y = 0.0;
                    pose.pose.orientation.z = 0.0;
                    pose.pose.orientation.w = 1.0;
                    cross_pub_.publish(pose);
                    last_pose_x = pose.pose.position.x;
                    last_pose_y = pose.pose.position.y;
                    last_pose=0;
                } else {
                    last_pose=10;
                }
            } else {
                last_pose=0;
                last_pose_x = pose.pose.position.x;
                last_pose_y = pose.pose.position.y;
            }
            if (unknown_alt) {
                msg.altitude = 0.0;
                msg.accuracy = 0.0;
            } else {
                msg.altitude = altitude+cross_expected_altitude_ + 0.18;
                msg.accuracy = 0.01*(altitude-3.0)*(altitude-3.0)+0.1;
            }
        } else {
            msg.altitude = 0.0;
            msg.accuracy = 0.0;
        }
        altitude_pub_.publish(msg);
    } else {
        // if the target was not detected, publish empty message
        cross_detector::cross_alt msg;
        msg.header.stamp = image_msg->header.stamp;
        msg.altitude = 0.0;
        msg.accuracy = 0.0;
        altitude_pub_.publish(msg);
        if (last_pose<10) {
            last_pose++;
        }
    }
    
    // DEBUG FRAME RATE AND DETECTION RATE
    if (fps_cnt++ > 100) {
        ros::Duration interval = ros::Time::now() - fps_time_old;
        ROS_INFO("Actual %f FPS detection rate: %f ", fps_cnt/interval.toSec(), det_cnt/(double)fps_cnt);
        fps_cnt = 0;
        det_cnt = 0;
        fps_time_old = ros::Time::now();
    }
}

/**
 * @brief main
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv)
{
    ros::init(argc, argv, "cross_detector");
    CrossDetector cd;
    
    ros::spin();
    return 0;
}
