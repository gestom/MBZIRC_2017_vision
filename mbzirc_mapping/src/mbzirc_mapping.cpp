#include <ros/ros.h>
#include <ros/package.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <ctime>
#include <math.h>
#include <time.h>
#include <queue>

#include <object_detection/detectedobject.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/Pose.h>
#include <mbzirc_mapping/GetObjects.h>
#include <std_srvs/Trigger.h>

#define MAX_OBJECTS 1000
#define NUM_TYPES 10

typedef struct 
{
  float x, y, d;
  int type, points, lastFrame;
  int numObservations;
  float likes[NUM_TYPES];
  float fx, fy, fd;
}SObject;

bool debug = true;
SObject objectArray[MAX_OBJECTS];
int numObjects = 0;
double minDistance = 5.0;
int minObservations = 5;

ros::Timer mapPublishTimer; 
ros::ServiceServer service_get_small_object;
ros::ServiceServer service_get_large_object;
ros::ServiceServer service_reset_estimation;

/*** UPDATE STEP OF COLOR TARGET MAP ***/
void objectCallback(const object_detection::detectedobject::ConstPtr &object){

  // object->type
  // 0 --- centroid of large object
  // 1 --- Red static object
  // 2 --- Green static object
  // 3 --- Blue static object
  // 5 --- moving objects
  // 10+ --- corners of large object

  //moving objects are ignored
  if (object->type == 5 || object->type >= NUM_TYPES) return;
  int type = object->type;

  /**5.1 the closest object from M to d **/
  float minimalDistance = 1000;
  int index = -1;
  for (int i = 0;i<numObjects;i++){

    float dx = objectArray[i].fx-object->absolute.position.x;	
    float dy = objectArray[i].fy-object->absolute.position.y;

    if (sqrt(dx*dx+dy*dy) < minimalDistance && objectArray[i].type == type && objectArray[i].lastFrame < object->imageID){
      index = i;
      minimalDistance = sqrt(dx*dx+dy*dy);
    }
  }

  if (debug){
    ROS_INFO("Object %i %i %i %.2lf %.2lf %.2f",
        object->imageID,
        object->id,
        object->type,
        object->absolute.position.x,
        object->absolute.position.y,
        minimalDistance);
  }

  /**5.2 if the object was seen before **/
  if (minimalDistance < minDistance) 
  {
	  //**5.3 update its position **/
	  objectArray[index].x += object->absolute.position.x;
	  objectArray[index].y += object->absolute.position.y;
	  objectArray[index].d += minimalDistance;
	  objectArray[index].lastFrame = object->imageID;

	  //**5.4 + 5.5 update color and score (the same for round objects) **/ 
	  if (object->type >= 0 && object->type < NUM_TYPES) objectArray[index].likes[object->type]++; //atlernatively, classification likelyhood

	  //**5.6 update the most likely score (type) **/
	  float maxLike = 0;
	  int typeIndex = 0;
	  for (int j=0;j<NUM_TYPES;j++){
		  if (objectArray[index].likes[j] > maxLike){
			  maxLike = objectArray[index].likes[j];
			  typeIndex = j;
		  }
	  }
	  objectArray[index].type = typeIndex;

	  //**5.7 increment the number of observations **/
	  objectArray[index].numObservations++;

	  /*recalculate object positions*/
	  objectArray[index].fx = objectArray[index].x / objectArray[index].numObservations;
	  objectArray[index].fy = objectArray[index].y / objectArray[index].numObservations;
	  objectArray[index].fd = objectArray[index].d / objectArray[index].numObservations;
  }
  else if ( numObjects < MAX_OBJECTS)
  {
	  /**5.8 add detected object to the map**/
	  objectArray[numObjects].x = object->absolute.position.x;
	  objectArray[numObjects].y = object->absolute.position.y;
	  objectArray[numObjects].numObservations = 1;
	  objectArray[numObjects].d = 0.1;
	  objectArray[numObjects].type = type;
	  objectArray[numObjects].lastFrame = object->imageID;

	  /*initialise likelyhoods of object type (score)*/ 
	  memset(objectArray[numObjects].likes,0,sizeof(float)*NUM_TYPES);
	  if (object->type >= 0 && object->type < NUM_TYPES) objectArray[numObjects].likes[object->type]=1; //atlernatively, classification likelyhood

	  /*recalculate object data for output*/
	  objectArray[numObjects].fx = objectArray[numObjects].x;
	  objectArray[numObjects].fy = objectArray[numObjects].y;
	  objectArray[numObjects].fd = objectArray[numObjects].d;

	  numObjects++;

  }

}

// service callback for sending estimated positions of small objects 
bool smallObjectsSrvCB(mbzirc_mapping::GetObjects::Request &req, mbzirc_mapping::GetObjects::Response &res){

  geometry_msgs::PoseWithCovariance pose;
  pose.pose.orientation.w = 1;

  for (int i = 0;i<numObjects;i++)
  {
    if (objectArray[i].numObservations > minObservations){
      pose.pose.position.x = objectArray[i].fx;
      pose.pose.position.y = objectArray[i].fy;
      if (objectArray[i].type >  0 && objectArray[i].type < 4) res.poses.push_back(pose); 
    }
  }
  res.success = true;
  return true;
}

// service callback for sending estimated positions of large objects 
bool largeObjectsSrvCB(mbzirc_mapping::GetObjects::Request &req, mbzirc_mapping::GetObjects::Response &res)
{

  geometry_msgs::PoseWithCovariance pose;
  pose.pose.orientation.w = 1;

  for (int i = 0;i<numObjects;i++)
  {
    if (objectArray[i].numObservations > minObservations){
      pose.pose.position.x = objectArray[i].fx;
      pose.pose.position.y = objectArray[i].fy;
      if (objectArray[i].type == 0) 
        res.poses.push_back(pose); 
    }
  }
  res.success = true;
  return true;
}

// service callback for reseting estimation
bool resetEstimationSrvCB(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res)
{
	ROS_INFO("Reseting estimation.");
	numObjects = 0;

	res.success = true;
	res.message = "Estimation reseted.";
	return true;
}

int main(int argc, char** argv) 
{
	// Initialize ROS Node
	ros::init(argc, argv, "mbzirc_mapping");
	ROS_INFO("%s: Starting", ros::this_node::getName().c_str());
	ros::NodeHandle nh("~");

	nh.param("minObservations", minObservations, 5);
	nh.param("debug", debug, false);
	nh.param("minDistance", minDistance, 5.0);

	ros::Subscriber subObjects = nh.subscribe("objects", 1, &objectCallback, ros::TransportHints().tcpNoDelay());

	service_get_small_object = nh.advertiseService("get_small_objects", smallObjectsSrvCB);
	service_get_large_object = nh.advertiseService("get_large_objects", largeObjectsSrvCB);
	service_reset_estimation = nh.advertiseService("reset_estimation", resetEstimationSrvCB);

	ros::spin();

	return 0;
}
