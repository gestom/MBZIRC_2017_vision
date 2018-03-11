#include <signal.h>
#include <ros/ros.h>
#include "CTimer.h"
#include "CSegmentation.h"
#include "CTransformation.h"
#include <image_transport/image_transport.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Int32.h>
#include <dynamic_reconfigure/server.h>
#include <object_detection/object_detectionConfig.h>
#include <object_detection/detectedobject.h>
#include <object_detection/ObjectWithType.h>
#include <object_detection/ObjectWithTypeArray.h>
#include <ros/package.h>
#include <tf/transform_listener.h>
#include "opencv2/ml/ml.hpp"

#include <sys/time.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <string>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>

using namespace std;
using namespace cv;

VideoCapture *capture;

string uav_name;

String colorMap;
int objectImageSequence = 0;
int imageNumber = 0;
Mat gaussianSamples;
Mat storedSamples;
ros::Time lastTime;
geometry_msgs::PoseStamped lastPose;
geometry_msgs::Point diffPose;
geometry_msgs::Point offset;

MatND histogram;
int hbins = 180;
int sbins = 256;
tf::TransformListener *listener;
tf::StampedTransform lastTransform;
bool gui = true;
bool debug = false;
bool oldbags = false;
bool stallImage = false;
int beginX = 0;
int beginY = 0;
ros::Publisher poseArrayPub;
ros::Publisher objectWithTypeArrayPub;
geometry_msgs::PoseArray poseArray;

image_transport::Publisher imdebug;
image_transport::Publisher objectImages;
image_transport::Publisher imhist;
image_transport::Publisher frame_pub;
ros::Publisher pose_pub;
ros::Publisher errorPub;
ros::Publisher objectPublisher;
ros::Subscriber localOdomSub;

SSegment currentSegment;
SSegment lastSegment;

CSegmentation segmentation;
CTransformation *altTransform;
Mat videoFrame,frama,distorted;
Mat imageCoords,metricCoords;
CTimer timer;
int frameExists = 0;
int detectedObjects = 0;
Mat mask,frame,rframe;
SSegment segments[MAX_SEGMENTS];
object_detection::detectedobject objectDescriptionArray[MAX_SEGMENTS];

/* filled from the config file */
Mat distCoeffs = Mat_<double>(1,5);	
Mat intrinsic = Mat_<double>(3,3);	

cv_bridge::CvImage cv_ptr;

int defaultImageWidth= 640;
int defaultImageHeight = 480;
int minSegmentSize = 10;
int maxSegmentSize = 1000000;
float distanceTolerance = 0.2;
float outerDimUser = 0.05;
float outerDimMaster = 0.07;
double camera_yaw_offset = 0;
double camera_phi_offset = 0;
double camera_psi_offset = 0;
double camera_delay = 0.22;
double object_height = 0.20;
double camera_offset = 0.17;
double circleDiameter = 0.2;
int histogramScale = 3;
float visualDistanceToleranceRatio = 0.2;
float visualDistanceToleranceAbsolute =  0.2;

std::string colormap_filename;

//parameter reconfiguration
void reconfigureCallback(object_detection::object_detectionConfig &config, uint32_t level) 
{
  maxSegmentSize = config.maxBlobSize;
  minSegmentSize = config.minBlobSize;
  segmentation.minRoundness = config.minRoundness;
  segmentation.minCircularity = config.minCircularity;
  circleDiameter = config.objectDiameter;
  histogramScale = config.histogramScale;
  visualDistanceToleranceRatio = config.visualDistanceToleranceRatio;
  visualDistanceToleranceAbsolute = config.visualDistanceToleranceAbsolute;
  //outerDimMaster = config.masterDiameter/100.0;
  //distanceTolerance = config.distanceTolerance/100.0;
  //detector->reconfigure(config.initialRoundnessTolerance, config.finalRoundnessTolerance, config.areaRatioTolerance,config.centerDistanceToleranceRatio,config.centerDistanceToleranceAbs);
}

void learnSegments(int number);

void graspCallback(const std_msgs::Int32 &msg)
{
  if (msg.data == -1 || msg.data == -2){
    offset.x = diffPose.x*0.5;
    offset.y = diffPose.y*0.5;
  }else{
    offset.x = 0;
    offset.y = 0;
  }
}

void saveColors()
{
	if (detectedObjects == 4){
		float ix[] = {-1,-0,+1,+0};
		float iy[] = {+0,-1,-0,+1};
		int ii[4];
		string filename = ros::package::getPath("object_detection")+"/etc/"+colormap_filename+".col";
		FILE *file = fopen(filename.c_str(),"w+");
		for (int i = 0;i<detectedObjects;i++)
		{
			int index = 0;
			float minEval = -100000;
			float eval = -100000;
			for (int j = 0;j<4;j++){
				eval = 	objectDescriptionArray[j].x*ix[i] + objectDescriptionArray[j].y*iy[i];
				if (eval > minEval){
					minEval = eval;
					index = j;
				}
			}
			int ii = i;
			if (ii == 0) ii = 4;
			fprintf(file,"%i %.0f %.0f %.0f\n",ii,objectDescriptionArray[index].h,objectDescriptionArray[index].s,objectDescriptionArray[index].v);
			segmentation.setColor(ii,objectDescriptionArray[index].h,objectDescriptionArray[index].s,objectDescriptionArray[index].v);
		}
		fclose(file);
	}
}

/**** HERE IS WHERE THE IMAGE PROCESSING HAPPENS ****/
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	segmentation.statTotalSegments=segmentation.statGoodSizeSegments=segmentation.statCircularSegments=segmentation.statRoundSegments=segmentation.statFinalSegments=0;
	if (msg->header.stamp.sec < 1) return;

	/** first, determine drone altitude, which is needed to check the expected size of objects -- this is done by transforming 0,0,0 from local UAV frame to the global frame **/
	geometry_msgs::PoseStamped pose;
	geometry_msgs::PoseStamped tf_pose;
	geometry_msgs::PoseStamped dronePose;
	debug = false;
	if (debug) {
		poseArray.poses.clear();
		poseArray.header.frame_id = string("/fcu_")+uav_name;
		if (oldbags)
			poseArray.header.frame_id = "/fcu";
	}

	/* fill the 0,0,0 and transform */
	float az = 0;
	try {
		pose.header.frame_id = string("/fcu_")+uav_name;
		if (oldbags) pose.header.frame_id = "/fcu";
		pose.header.stamp = msg->header.stamp-ros::Duration(camera_delay);
		pose.pose.position.x = 0;
		pose.pose.position.y = 0;
		pose.pose.position.z = 0;
		pose.pose.orientation.x = 0;
		pose.pose.orientation.y = 0;
		pose.pose.orientation.z = 0;
		pose.pose.orientation.w = 1;
		listener->transformPose("/local_origin",pose,dronePose);
	}
	catch (tf::TransformException &ex) {
		ROS_ERROR("%s",ex.what());
	}

	ros::Duration rozdil = ros::Time::now() - lastTime;
	lastTime = ros::Time::now();

	/* store UAV attitude and altitude */ 
	double roll, pitch, yaw;
	tf::Quaternion attitude(dronePose.pose.orientation.x,dronePose.pose.orientation.y,dronePose.pose.orientation.z,dronePose.pose.orientation.w);
	tf::Matrix3x3 m(attitude);
	az = dronePose.pose.position.z;

	if (debug) printf("%06d T0 %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",imageNumber,rozdil.sec*1000000000+rozdil.nsec,dronePose.pose.position.x,dronePose.pose.position.y,dronePose.pose.position.z,dronePose.pose.orientation.x,dronePose.pose.orientation.y,dronePose.pose.orientation.z,dronePose.pose.orientation.w);

	if (stallImage == false) frame = cv_bridge::toCvShare(msg, "bgr8")->image;

	/** perform the segmentation **/
	timer.reset();
	timer.start();

	/** SEGMENTATION ITSELF - ALGORITHMS 3 and 4 of the paper**/
	segmentation.findSegment(&frame,&imageCoords,segments,minSegmentSize,maxSegmentSize);
	altTransform = NULL;

	//calculate the relative position of the object (without estimating the distance
	if (imageCoords.rows!=0){
		undistortPoints(imageCoords,metricCoords,intrinsic,distCoeffs);
		altTransform = new CTransformation(intrinsic,distCoeffs,circleDiameter);
	}

	//draw results in the image and publish the detected objects' description and images
	object_detection::detectedobject objectDescription;
	object_detection::ObjectWithTypeArray object_with_type_array;

	if (debug) {
		poseArray.header.seq = objectImageSequence;
	}

	detectedObjects = 0;

	/** get all the detected objects from ALGORITHM 4**/
	for(int i = 0; i < imageCoords.rows; i++)
	{
		/** verification of the object closeness to the ground plane **/
		//calculate object position as if on the ground
		char description[1000];
		float rx = az*metricCoords.at<float>(i,0);
		float ry = az*metricCoords.at<float>(i,1); 
		float d = sqrt(rx*rx+ry*ry+az*az); 
		float expected = segments[i].size*d*d;
		objectDescription.cornerX.clear();
		objectDescription.cornerY.clear();

		/**calculation of the object position, see section 3.5**/
		STrackedObject o = altTransform->transform(segments[i]);
		float od = sqrt(o.x*o.x+o.y*o.y+o.z*o.z); 
		if (d == 0) d = 0.0001;
		/** test object distance from the ground plane, see end of 3.5 **/
		if (fabs(o.x/d-1.0) < visualDistanceToleranceRatio || fabs(od - d) < visualDistanceToleranceAbsolute){
					segmentation.statFinalSegments++; 

					cout << imageNumber<<" object "<<rx<<" "<<ry<<" "<<az<<" "<< d <<" size "<< segments[i].size << " expected " <<expected<<" circularity "<<segments[i].circularity << " Trans: " << o.x << " " << o.y << " " << o.z << " Type: " << segmentation.classifySegment(segments[i]) <<endl;

					//fill detected object statistics for publication
					pose.pose.position.x = o.x;
					pose.pose.position.y = o.y;
					pose.pose.position.z = o.z;
					pose.pose.orientation.x = 0.0;
					pose.pose.orientation.y = 0.0;
					pose.pose.orientation.z = 0.0;
					pose.pose.orientation.w = 1.0;
					objectDescription.visual = pose.pose;

					objectDescription.full = !segments[i].warning;
					objectDescription.x = segments[i].x;
					objectDescription.y = segments[i].y;
					objectDescription.h = segments[i].h;
					objectDescription.s = segments[i].s;
					objectDescription.v = segments[i].v;
					objectDescription.size = segments[i].size;
					objectDescription.minX = segments[i].minX;
					objectDescription.minY = segments[i].minY;
					objectDescription.maxX = segments[i].maxX;
					objectDescription.maxY = segments[i].maxY;
					objectDescription.type = segmentation.classifySegment(segments[i]);
					objectDescription.id = objectImageSequence++; 
					objectDescription.imageID = imageNumber; 
					objectDescription.px = rx; 
					objectDescription.py = ry; 
					objectDescription.pz = az;
					objectDescription.d = d; 
					for (int ii = 0;ii<segments[i].contourPoints;ii++)
					{
						objectDescription.contourX.push_back(segments[i].contourX[ii]);
						objectDescription.contourY.push_back(segments[i].contourY[ii]);
					}
					for (int ii = 0;ii<4;ii++)
					{
						objectDescription.cornerX.push_back(segments[i].cornerX[ii]);
						objectDescription.cornerY.push_back(segments[i].cornerY[ii]);
					}
					objectDescriptionArray[detectedObjects++] = objectDescription;

					//draw results in the complete image 
					if (gui) {
						Scalar color;
						if (objectDescription.type == 1) color = Scalar(0,0,255);
						if (objectDescription.type == 2) color = Scalar(0,255,0);
						if (objectDescription.type == 3) color = Scalar(255,0,0);
						if (objectDescription.type == 5) color = Scalar(0,255,255);
						circle(frame,cvPoint(imageCoords.at<float>(i,0),imageCoords.at<float>(i,1)),sqrt(segments[i].size),color,3,8,0);
						line(frame,cvPoint(imageCoords.at<float>(i,0)+sqrt(segments[i].size/2),imageCoords.at<float>(i,1)-sqrt(segments[i].size/2)),cvPoint(imageCoords.at<float>(i,0)+35,imageCoords.at<float>(i,1)-35),color,2,0);
						line(frame,cvPoint(imageCoords.at<float>(i,0)+35,imageCoords.at<float>(i,1)-35),cvPoint(imageCoords.at<float>(i,0)+245,imageCoords.at<float>(i,1)-35),color,2,0);
						sprintf(description,"Pos: %.2f %.2f %.2f",rx,ry,az);
						//sprintf(description,"Pos: %.2f %.2f %.2f",o.x,od,d);
						putText(frame,description,cvPoint(imageCoords.at<float>(i,0)+45,imageCoords.at<float>(i,1)-40), CV_FONT_HERSHEY_PLAIN, 1.1,Scalar(0,0,0),2);
						sprintf(description,"Attrs: %.3f %.3f %i",segments[i].roundness,segments[i].circularity,segments[i].size);
						putText(frame,description,cvPoint(imageCoords.at<float>(i,0)+45,imageCoords.at<float>(i,1)-20), CV_FONT_HERSHEY_PLAIN, 1.1,Scalar(0,0,0),2);
					}
			}
	}//putText(img, text, textOrg, fontFace, fontScale,  Scalar::all(255), thickness, 8);

	printf("Segmentation %i %i %i %i %i %i\n",segmentation.statTotalSegments,segmentation.statGoodSizeSegments,segmentation.statRoundSegments,segmentation.statCircularSegments,segmentation.statFinalSegments,detectedObjects);

	if (altTransform != NULL) delete altTransform;
	if (debug) {

		frame.copyTo(videoFrame);
		cv_ptr.encoding = "bgr8";
		cv_ptr.image = videoFrame;
		sensor_msgs::ImagePtr imagePtr2 = cv_ptr.toImageMsg();
		imagePtr2->header.seq = cv_bridge::toCvShare(msg, "bgr8")->header.seq;

		try {
			frame_pub.publish(imagePtr2);
		} catch (...) {
			ROS_ERROR("Exception caught during publishing topic %s.", frame_pub.getTopic().c_str());
		}
	}

	//END of segmentation - START calculating global frame coords
	if (gui) {
		imshow("frame",frame);
		char dummy[1000];
		sprintf(dummy,"%04i.jpg",imageNumber);
		imwrite(dummy,frame);
	}

	//calculate global coords of the objects
	for (int i = 0; i < detectedObjects; i++)
	{

		pose.header.frame_id = string("/fcu_")+uav_name;
		if (oldbags) pose.header.frame_id = string("/fcu");
		pose.header.stamp = msg->header.stamp-ros::Duration(camera_delay);
		objectDescription = objectDescriptionArray[i];
		float alpha = M_PI/2 + camera_yaw_offset;
		float phi =  + camera_phi_offset;;
		float psi =  + camera_psi_offset;;
		float x = objectDescription.px;
		float y = objectDescription.py;
		float z = objectDescription.pz;

		objectDescription.timestamp = pose.header.stamp;

		pose.pose.position.x = cos(alpha)*x-sin(alpha)*y;
		pose.pose.position.y = sin(alpha)*x+cos(alpha)*y;
		pose.pose.position.z = z;
		x = pose.pose.position.x;
		y = pose.pose.position.y;
		z = pose.pose.position.z;

		pose.pose.position.x = cos(psi)*x-sin(psi)*z;
		pose.pose.position.y = y;
		pose.pose.position.z = sin(psi)*x+cos(psi)*z;
		x = pose.pose.position.x;
		y = pose.pose.position.y;
		z = pose.pose.position.z;

		pose.pose.position.x = x;
		pose.pose.position.y = cos(phi)*y-sin(phi)*z;
		pose.pose.position.z = sin(phi)*y+cos(phi)*z;

		pose.pose.position.x += camera_offset;
		pose.pose.position.y = -pose.pose.position.y; // - 0.10;
		pose.pose.position.z = -pose.pose.position.z;

		pose.pose.orientation.x = 0.707;
		pose.pose.orientation.y = 0.0;
		pose.pose.orientation.z = 0.0;
		pose.pose.orientation.w = 0.707;
		objectDescription.relative = pose.pose;

		try {
			listener->transformPose("/local_origin",pose,tf_pose);
		}
		catch (tf::TransformException &ex) {
			ROS_ERROR("%s",ex.what());
			return;
			//ros::Duration(1.0).sleep();
		}

		if (debug) {
			printf("%06d R%i %d %.3f %.3f %.3f %.3f %.3f %.3f\n",imageNumber,1+objectDescription.type,rozdil.sec*1000000000+rozdil.nsec,pose.pose.position.x,pose.pose.position.y,pose.pose.position.z,objectDescription.h,objectDescription.s,objectDescription.v);
			printf("%06d T%i %d %.3f %.3f %.3f %.3f %.3f %.3f\n",imageNumber,1+objectDescription.type,rozdil.sec*1000000000+rozdil.nsec,tf_pose.pose.position.x,tf_pose.pose.position.y,tf_pose.pose.position.z,objectDescription.h,objectDescription.s,objectDescription.v);
		}
		float height_correction = object_height;
		if (objectDescription.type == 5) height_correction = 0.42; 

		float ratio = dronePose.pose.position.z/(dronePose.pose.position.z-tf_pose.pose.position.z+height_correction);
		pose.pose.position.x -= camera_offset;
		pose.pose.position.x *= ratio;
		pose.pose.position.x += camera_offset;
		pose.pose.position.y *= ratio;
		pose.pose.position.z *= ratio;
		//			cout << imageNumber<<" Compare "<<pose.pose.position.x<<" "<<pose.pose.position.y<<" "<< pose.pose.position.z<<" " << " Trans: " << objectDescription.visual.position.x << " " << objectDescription.visual.position.y << " " << objectDescription.visual.position.z <<endl;
		try {
			listener->transformPose("/local_origin",pose,tf_pose);
		}
		catch (tf::TransformException &ex) {
			ROS_ERROR("%s",ex.what());
			return;
		}

		if (debug){
			printf("%06d C%i %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
					imageNumber,
					1+objectDescription.type,
					rozdil.sec*1000000000+rozdil.nsec,
					tf_pose.pose.position.x,
					tf_pose.pose.position.y,
					tf_pose.pose.position.z,
					objectDescription.h,
					objectDescription.s,
					objectDescription.v,
					dronePose.pose.position.z
			      );
		}
		objectDescription.uncorrected = objectDescription.absolute = tf_pose.pose;
		lastPose.pose.position.x = tf_pose.pose.position.x*0.2+lastPose.pose.position.x*0.8;
		lastPose.pose.position.y = tf_pose.pose.position.y*0.2+lastPose.pose.position.y*0.8;
		lastPose.pose.position.z = tf_pose.pose.position.z*0.2+lastPose.pose.position.z*0.8;
		lastPose = tf_pose;

		objectDescription.absolute.position.x = tf_pose.pose.position.x;
		objectDescription.absolute.position.y = tf_pose.pose.position.y;
		
		// publish topics
		try {
			objectPublisher.publish(objectDescription); // neede for mapping
		} catch (...) {
			ROS_ERROR("Exception caught during publishing topic %s.", objectPublisher.getTopic().c_str());
		}

		// publish array of poses with type, needed byt landing object estimator
		object_detection::ObjectWithType object_with_type;
		object_with_type.x = tf_pose.pose.position.x+offset.x;
		object_with_type.y = tf_pose.pose.position.y+offset.y;
		object_with_type.type = objectDescription.type;
		object_with_type.z = objectDescription.visual.position.x;

		object_with_type.stamp = ros::Time::now();

		object_with_type_array.objects.push_back(object_with_type); // needed for estimation
		object_with_type_array.stamp = ros::Time::now();

		poseArray.header.frame_id = "local_origin";
		poseArray.poses.push_back(tf_pose.pose);
	}

	diffPose.x = lastPose.pose.position.x-dronePose.pose.position.x;
	diffPose.y = lastPose.pose.position.y-dronePose.pose.position.y;
	diffPose.z = dronePose.pose.position.z;
	//errorPub.publish(diffPose);
	if (debug){
			printf("%06d D%i %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
					imageNumber,
					1+objectDescription.type,
					rozdil.sec*1000000000+rozdil.nsec,
					lastPose.pose.position.x-dronePose.pose.position.x,
					lastPose.pose.position.y-dronePose.pose.position.y,
					lastPose.pose.position.z-dronePose.pose.position.z,
					objectDescription.h,
					objectDescription.s,
					objectDescription.v,
					dronePose.pose.position.z
			      );
			printf("%06d O%i %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
					imageNumber,
					1+objectDescription.type,
					rozdil.sec*1000000000+rozdil.nsec,
					lastPose.pose.position.x-dronePose.pose.position.x-offset.x,
					lastPose.pose.position.y-dronePose.pose.position.y-offset.y,
					lastPose.pose.position.z-dronePose.pose.position.z,
					objectDescription.h,
					objectDescription.s,
					objectDescription.v,
					dronePose.pose.position.z
			      );

		}
	if (debug) {
		try {
			poseArrayPub.publish(poseArray);
		} catch (...) {
			ROS_ERROR("Exception caught during publishing topic %s.", poseArrayPub.getTopic().c_str());
		}
	}

	try {
		objectWithTypeArrayPub.publish(object_with_type_array);
	} catch (...) {
		ROS_ERROR("Exception caught during publishing topic %s.", objectWithTypeArrayPub.getTopic().c_str());
	}

	/*processing user input*/
	int key = waitKey(1)%256;
	if (key == 32) stallImage = !stallImage;
	if (key == 'r'){
		segmentation.resetColorMap();
		histogram = Mat::zeros(hbins,sbins,CV_32FC1);
		storedSamples = Mat::zeros(0,3,CV_32FC1);
	}
	if (key == 's') segmentation.saveColorMap(colorMap.c_str());
	if (key == 'c') saveColors();
	if (key >= '1' && key < '9') learnSegments(key-'0');
	imageNumber++;
}

void learnNegative()
{

}

void learnSegments(int number)
{
/*  Mat sample = Mat::zeros(1,3,CV_32FC1);
  Mat histImg = Mat::zeros(180,255, CV_8UC3);

  EM em(number,EM::COV_MAT_DIAGONAL);
  Mat labels;
  printf("Training start\n");
  Mat means = Mat::zeros(3,3, CV_32FC1);
  means.at<float>(0,0) = 90;
  means.at<float>(0,1) = 128;
  means.at<float>(0,2) = 128;
  means.at<float>(1,0) = 170;
  means.at<float>(1,1) = 200;
  means.at<float>(1,2) = 128;
  means.at<float>(2,0) = 10;
  means.at<float>(2,1) = 200;
  means.at<float>(2,2) = 128;
  printf("Training progress %i %i\n",means.rows,means.cols);

  //em.trainE(storedSamples, means);
  em.train(storedSamples);
  printf("Training finished\n");
  Vec2d pred;
  for( int h = 0; h < 180; h+=4 ){
    for( int s = 0; s < 256; s+=4 )
    {
      for( int v = 0; v < 256; v+=4 )
      {
        float binVal = histogram.at<float>(h, s);
        sample.at<float>(0) = h;
        sample.at<float>(1) = s;
        sample.at<float>(2) = v;
        Vec2d pred = em.predict(sample);
        Vec3b color;
        if (pred[1] == 1.0) color = Vec3b(1,0,0);
        if (pred[1] == 2.0) color = Vec3b(0,0,1);
        if (pred[1] == 3.0) color = Vec3b(0,1,0);
        for (int ih = 0;ih <4;ih++){
          for (int is = 0;is <4;is++){
            histImg.at<Vec3b>(h+ih,s+is) =  histImg.at<Vec3b>(h+ih,s+is)+color;
          }
        }
      }
    }
    printf("Classify: %i\n",h);
  }
  Mat hsv(1,1,CV_8UC3);
  Mat rgb(1,1,CV_8UC3);
  Vec3b hsvp;
  Vec3b rgbp;
  int sampleType = 1;
  if (number == 1) sampleType = 0;	
  for (int r=0;r<64;r++){
    for (int g=0;g<64;g++){
      for (int b=0;b<64;b++){
        rgbp(0) = r*4;
        rgbp(1) = g*4;
        rgbp(2) = b*4;
        rgb.at<Vec3b>(0,0) = rgbp;
        cvtColor(rgb,hsv, COLOR_RGB2HSV,1);
        hsvp = hsv.at<Vec3b>(0,0);
        sample.at<float>(0) = hsvp(0);
        sample.at<float>(1) = hsvp(1);
        sample.at<float>(2) = hsvp(2);
        Vec2d pred = em.predict(sample);
        int i = (r*64+g)*64+b;
        if (pred[1] > 0.0){
          segmentation.colorArray[i] = sampleType;
          printf("PRED: %.3f\n",pred[0]);
        }else{
          segmentation.colorArray[i] = 1-sampleType;
        }
      }
    }
  }

  printf("Classification finished\n");

  if (gui) imshow("histogram",histImg);*/
}

void mainMouseCallback(int event, int x, int y, int flags, void* userdata)
{
  //start dragging - starting to define region of interrest
  if  ( event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN)
  {
    beginX = x;
    beginY = y;
  }
  //end dragging -  region of interrest defined
  if  ( (event == EVENT_LBUTTONUP || event == EVENT_LBUTTONUP))
  {
    //reverse positions of x/beginX and y/beginY if mouse drag in opposite direction 
    if (beginX > x){int tmpx = x;x=beginX;beginX = tmpx;}
    if (beginY > y){int tmpy = y;y=beginY;beginY = tmpy;}

    //cut off a region of interrest and convert to HSV color space
    Mat sample = Mat::zeros(1,3,CV_32FC1);
    Mat roi = frame(Rect(beginX,beginY ,x-beginX+1,y-beginY+1));
    Mat hsv(roi.rows,roi.cols,CV_8UC3);
    cvtColor(roi,hsv, COLOR_RGB2HSV,1);

    //create a list of pixels with given colors  
    for( int iy = 0; iy < hsv.rows; iy++ ){
      for( int ix = 0; ix < hsv.cols; ix++ ){
        Vec3b pixel = hsv.at<Vec3b>(iy,ix);
        for( int i = 0; i < 3;i++ ) sample.at<float>(i) = pixel(i);
        storedSamples.push_back(sample);
      }
    }

    //fill the histogram with the color samples   
    Mat histogram = Mat::zeros(180,256, CV_32FC1);
    for( int i = 0; i < storedSamples.rows; i++ ) histogram.at<float>((int)storedSamples.at<float>(i,0),(int)storedSamples.at<float>(i,1))++;

    //fill the histogram with the samples   
    double maxVal=0;
    histogram.at<float>(0,0)=0;
    minMaxLoc(histogram, 0, &maxVal, 0, 0);
    int scale = 1;
    Mat histImg = Mat::zeros(histogramScale*180, histogramScale*256, CV_8UC3);

    for( int h = 0; h < histogramScale*180; h++ ){
      for( int s = 0; s < histogramScale*256; s++ ){
        float binVal = histogram.at<float>(h/histogramScale, s/histogramScale);
        int intensity = cvRound(logf(binVal)*255/logf(maxVal));
        Vec3b pixel = Vec3b(intensity,intensity,intensity);
        histImg.at<Vec3b>(h,s) = pixel;
      }
    }

    if (gui) imshow("roi",roi);
    if (gui) imshow("histogram",histImg);

    cout << frame.at<Vec3b>(y,x) << endl;
  }
}

void histogramMouseCallback(int event, int x, int y, int flags, void* userdata)
{
  if  ( event == EVENT_LBUTTONDOWN )
  {
    beginX = x;
    beginY = y;
  }
  if  ( event == EVENT_LBUTTONUP )
  {
	  beginX = beginX/histogramScale;
	  beginY = beginY/histogramScale;
	  x=x/histogramScale;
	  y=y/histogramScale;
	  //segmentation.learnPixel(beginY,y,beginX,x,50,255);
	  segmentation.learnPixel(beginY-2,y+2,beginX-2,x+2,150,255);
	  //segmentation.learnPixel(0,180,0,255,200,255);
  }
}

//to speed up termination
void termHandler(int s){
  exit(1); 
}

int main(int argc, char** argv) 
{
	offset.x = offset.y = diffPose.x = diffPose.y = 0;

	ros::init(argc, argv, "object_detector");
	ros::NodeHandle n = ros::NodeHandle("~");

	n.param("uav_name", uav_name, string());
	n.param("gui", gui, false);
	n.param("debug", debug, false);
	if (gui) {
		debug = true;
		signal (SIGINT,termHandler);
	}

	if (uav_name.empty()) {

		ROS_ERROR("UAV_NAME is empty");
		ros::shutdown();
	}

	n.param("camera_yaw_offset", camera_yaw_offset, 0.0);
	n.param("camera_phi_offset", camera_phi_offset, 0.0);
	n.param("camera_psi_offset", camera_psi_offset, 0.0);
	n.param("max_segment_size", maxSegmentSize, 1000);
	n.param("camera_delay", camera_delay, 0.0);
	n.param("camera_offset", camera_offset, 0.17);
	n.param("object_height", object_height, 0.20);
	n.param("colormap_filename", colormap_filename, std::string("rosbag.bin"));
	std::vector<double> tempList;
	int tempIdx = 0;

	n.getParam("camera_distCoeffs", tempList);
	for (int i = 0; i < 5; i++) {
		distCoeffs.at<double>(i) = tempList[i];
	}

	n.getParam("camera_intrinsic", tempList);
	tempIdx = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			intrinsic.at<double>(i, j) = tempList[tempIdx++];
		}
	}

	ROS_INFO_STREAM("Node initiated with params: camera_offset " << camera_offset << ", maxSegmentSize = " << maxSegmentSize << ", camera_delay = " << camera_delay << ", colormap_filename = " << colormap_filename);

	ROS_INFO_STREAM("Distortion matrix: \n" << distCoeffs);
	ROS_INFO_STREAM("Intrinsic matrix: \n" << intrinsic);

	int aa = 0;
	histogram = Mat::zeros(hbins,sbins,CV_32FC1);
	gaussianSamples = Mat::zeros(0,2,CV_32FC1);
	storedSamples = Mat::zeros(0,3,CV_32FC1);

	if (gui) namedWindow("frame", CV_WINDOW_AUTOSIZE);
	if (gui) namedWindow("histogram", CV_WINDOW_AUTOSIZE);
	if (gui) namedWindow("roi", CV_WINDOW_AUTOSIZE);
	colorMap = ros::package::getPath("object_detection")+"/etc/"+colormap_filename;
	segmentation.loadColorMap(colorMap.c_str());
	segmentation.loadColors((colorMap+".col").c_str());
	String maskPath = ros::package::getPath("object_detection")+"/etc/mask.png";
	mask = imread(maskPath);
	image_transport::ImageTransport it(n);
	listener = new tf::TransformListener();

	// initialize dynamic reconfiguration feedback
	dynamic_reconfigure::Server<object_detection::object_detectionConfig> server;
	dynamic_reconfigure::Server<object_detection::object_detectionConfig>::CallbackType dynSer;
	dynSer = boost::bind(&reconfigureCallback, _1, _2);
	server.setCallback(dynSer);

	// SUBSCRIBERS
	image_transport::Subscriber image = it.subscribe("image_in", 1, &imageCallback);
	ros::Subscriber subGrasp = n.subscribe("grasping_result", 1, &graspCallback, ros::TransportHints().tcpNoDelay());

	// PUBLISHERS
	objectWithTypeArrayPub = n.advertise<object_detection::ObjectWithTypeArray>("object_array", 1); // landing object estimator needs it
	objectPublisher = n.advertise<object_detection::detectedobject>("objects", 1);          // mapping needs it
	errorPub = n.advertise<geometry_msgs::Pose>("error", 1);

	// Debugging PUBLISHERS
	if (debug) {
		poseArrayPub = n.advertise<geometry_msgs::PoseArray>("objectPositions", 1);
		pose_pub = n.advertise<geometry_msgs::PoseStamped>("objectRelative", 1);
		imdebug = it.advertise("processedimage", 1);
		objectImages = it.advertise("objectImages", 1);
		imhist = it.advertise("histogram", 1);
		frame_pub = it.advertise("frame", 1);
	}

	sensor_msgs::Image msg;
	if (gui) setMouseCallback("frame", mainMouseCallback, NULL);
	if (gui) setMouseCallback("histogram", histogramMouseCallback, NULL);

	timer.reset();
	timer.start();

	ros::spin();
}

