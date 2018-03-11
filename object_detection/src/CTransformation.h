/*
 * File name: CTransformation.h
 * Date:      2005/11/07 18:10
 * Author:    
 */

#ifndef __CTRANSFORMATION_H__
#define __CTRANSFORMATION_H__

#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include "CSegmentation.h"
#include <opencv2/opencv.hpp>

using namespace cv;

typedef struct{
	float x,y,z,d;
	float pitch,roll,yaw;
	float roundness;
	float bwratio;
	float error;
	float esterror;
	int ID;
}STrackedObject;

class CTransformation
{
	public:
		CTransformation(Mat intrinsic,Mat distCoeefs,float diam);
		~CTransformation();

		void transformXY(float *ix,float *iy);
		void updateParams(float a,float b, float c, float d);

		void unbarrel(unsigned char* src,unsigned char* dst);
		STrackedObject transform(SSegment segment);
		STrackedObject eigen(double data[]);

	private:
		Mat intrinsic;	
		Mat distCoeffs;	
		float trackedObjectDiameter;
};

#endif
/* end of CTransformation.h */
