#include <ros/ros.h>
#include "CTimer.h"
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Twist.h>
#include <dynamic_reconfigure/server.h>
#include <ros/package.h>
#include <tf/transform_listener.h>
#include <object_detection/detectedobject.h>
#include "CTimer.h"

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <signal.h>

using namespace std;

string uav_name;
int obj_type = 0;

tf::TransformListener *listener;
tf::StampedTransform lastTransform;
ros::Subscriber objectSub;

tf::StampedTransform transforms[1000000];
geometry_msgs::Pose poses[1000000];
geometry_msgs::Pose absols[1000000];
int numData = 0;
float checkTransforms();
float transform(float delay,float alpha,float phi,float psi,float distance,bool verbose);
float object_height = 0.2;
bool processing = false;
bool adding = false;
CTimer timer;

void termHandler(int s)
{
	objectSub.shutdown();
	processing = true;
	while (adding) usleep(100000);
	fprintf(stderr,"Processing %i data points.\n",numData);
	checkTransforms();
}

void objectCallback(const object_detection::detectedobject &msg)
{
	adding = true;
	if (msg.type == obj_type && processing ==false){
		tf::StampedTransform pose_tf;
		listener->waitForTransform("/local_origin","/fcu_"+uav_name,msg.timestamp, ros::Duration(2));
		listener->lookupTransform("/local_origin","/fcu_"+uav_name,msg.timestamp,pose_tf);
		poses[numData] = msg.relative;
		transforms[numData] = pose_tf;
		printf("%i\n",numData);
		numData++;
	}
	timer.reset();
	adding = false;
}

float checkTransforms()
{
	float minimDev = 10000;
	float dev = 0;
	float statespace = 40*40*40;
	float progress = 0;
	float range = 0.5;
	float step = 0.1;
	float delay = 0;
	float delayGain = 10;
	float distanceMin,alphaMin,phiMin,psiMin,delayMin;
	float distanceZero,alphaZero,phiZero,psiZero,delayZero;
	distanceZero=alphaZero=phiZero=psiZero=delayZero=0.0;
	for (float step = 0.1;step > 0.0001;step = step/10){
		range = step*5;
		float distance = 00;
//		float delay = 0.28;
		for (float delay = -range*delayGain+delayZero;delay<range*delayGain+delayZero;delay+=step*delayGain){
//			for (float distance = -range+distanceZero;distance<range+distanceZero;distance+=step){
				for (float alpha = -range+alphaZero;alpha<range+alphaZero;alpha+=step){
					for (float phi = -range+phiZero;phi<range+phiZero;phi+=step){
						for (float psi = -range+psiZero;psi<range+psiZero;psi+=step){
							dev = transform(delay,alpha,phi,psi,distance,false);
							if (dev < minimDev){
								minimDev = dev;
								alphaMin = alpha; 
								psiMin = psi; 
								phiMin = phi;
								delayMin = delay;
								distanceMin = distance;	
							}
							progress++;
						}
					}
				}
			}
//		}
		alphaZero = alphaMin; 
		psiZero = psiMin; 
		phiZero = phiMin; 
		delayZero = delayMin; 
		distanceZero = distanceMin; 
		printf("Current %.4f %.3f %.3f %.3f %.3f %.3f\n",minimDev,alphaMin,phiMin,psiMin,distanceMin,delayMin);
	}
	string filename = ros::package::getPath("object_detection")+"/cfg/"+uav_name+"_latest.yaml";
	FILE *file = fopen(filename.c_str(),"w+");
		
	fprintf(file,"camera_yaw_offset: %.3f\n\n",alphaMin);

	fprintf(file,"camera_psi_offset: %.3f\n\n",psiMin);

	fprintf(file,"camera_phi_offset: %.3f\n\n",phiMin);
	fclose(file);

	dev = transform(delayZero,alphaZero,phiZero,psiZero,distanceZero,true);
	ros::shutdown();
	return minimDev;
}

float transform(float delay,float alpha,float phi,float psi,float distance,bool verbose)
{
	float sX,sY,sZ,ssX,ssY,ssZ;
	geometry_msgs::PoseStamped pose;
	sX=sY=sZ=ssX=ssY=ssZ = 0.0;
	int backPad = 10;
	int frontPad = 10;
	for (int i = frontPad;i<numData-backPad;i++){
		pose.pose = poses[i];
		float camera_offset = 0.12;
		float x = pose.pose.position.x - camera_offset;;
		float y = -pose.pose.position.y;
		float z = -pose.pose.position.z;

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

		x = pose.pose.position.x + camera_offset+distance;
		y = -pose.pose.position.y;
		z = -pose.pose.position.z;
		int ndelay = floor(delay);
		float fdelay = delay-ndelay;
		tf::Vector3 np = transforms[i-ndelay]*tf::Vector3(x,y,z);
		tf::Vector3 dp = transforms[i-ndelay]*tf::Vector3(0,0,0);
		tf::Vector3 np1 = transforms[i-ndelay-1]*tf::Vector3(x,y,z);
		tf::Vector3 dp1 = transforms[i-ndelay-1]*tf::Vector3(0,0,0);
		dp = dp1*fdelay + (1-fdelay)*dp;
		np = np1*fdelay + (1-fdelay)*np;
		float ratio = dp.getZ()/(dp.getZ()-np.getZ()+object_height);
		x *= ratio;
		y *= ratio;
		z *= ratio;
		np = transforms[i-ndelay]*tf::Vector3(x,y,z);
		np1 = transforms[i-ndelay-1]*tf::Vector3(x,y,z);
		np = np1*fdelay + (1-fdelay)*np;

		//pose.pose.position.x *= ratio;
		//pose.pose.position.y *= ratio;
		//pose.pose.position.z *= ratio;
		sX += np.getX();
		sY += np.getY();
		sZ += np.getZ();
		ssX += np.getX()*np.getX();
		ssY += np.getY()*np.getY();
		ssZ += np.getZ()*np.getZ();
//		if (verbose) printf("%.3f %.3f %.3f\n",np.getX(),np.getY(),np.getZ());
	}
	int numDat = numData-backPad-frontPad;
	return  sqrt(ssX/numDat-sX*sX/numDat/numDat)+sqrt(ssY/numDat-sY*sY/numDat/numDat)+sqrt(ssZ/numDat-sZ*sZ/numDat/numDat);
	//printf("%.3f %.3f %.3f %.3f %.3f %.3f\n",alpha,phi,psi,sX/numData,sY/numData,sZ/numData);
	//printf("%.3f %.3f %.3f %.3f %.3f %.3f\n",alpha,phi,psi,sqrt(ssX/numData-sX*sX/numData/numData),sqrt(ssY/numData-sY*sY/numData/numData),sqrt(ssZ/numData-sZ*sZ/numData/numData));
	//printf("Current %.3f %.3f %.3f %.3f\n",progress/statespace,alphaMin,phiMin,psiMin);
	//printf("%.3f %.3f %.3f\n",alphaMin,phiMin,psiMin);
}

int main(int argc, char** argv) 
{

	ros::init(argc, argv, "object_calibration");
	ros::NodeHandle n = ros::NodeHandle("~");
	signal (SIGINT,termHandler);

	n.param("uav_name", uav_name, string());
	n.param("obj_type", obj_type, int());
	printf("Name: %s %i\n",uav_name.c_str(),obj_type);
	objectSub = n.subscribe("/"+uav_name+"/mbzirc_detector/objects", 1, &objectCallback, ros::TransportHints().tcpNoDelay());
	listener = new tf::TransformListener();
	timer.reset();
	timer.start();
	while (ros::ok()){
		ros::spinOnce();
		if (timer.getTime() > 1000 && numData > 100) termHandler(0); 
	}
}

