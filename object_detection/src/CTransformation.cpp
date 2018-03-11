#include "CTransformation.h"
#include <stdio.h>
#include "sysmat.cpp" 

CTransformation::CTransformation(Mat intr,Mat dC,float diam)
{
	intrinsic = intr;
	distCoeffs = dC;
	trackedObjectDiameter = diam;
}

CTransformation::~CTransformation()
{
}

void CTransformation::transformXY(float *ax,float *ay)
{
	
	Mat coords = Mat::ones(1, 1, CV_32FC2);
	Mat metric;
	coords.at<float>(0) = *ax;
	coords.at<float>(1) = *ay;
	undistortPoints(coords,metric,intrinsic,distCoeffs);
	*ax = metric.at<float>(0,0);
	*ay = metric.at<float>(0,1);
}

STrackedObject CTransformation::eigen(double data[])
{
	STrackedObject result;
	result.error = 0;
	double d[3];
	double V[3][3];
	double dat[3][3];
	for (int i = 0;i<9;i++)dat[i/3][i%3] = data[i];
	eigen_decomposition(dat,V,d);

	//eigenvalues
	float L1 = d[1]; 
	float L2 = d[2];
	float L3 = d[0];
	//eigenvectors
	int V2=2;
	int V3=0;

	//detected pattern position
	float z = trackedObjectDiameter/sqrt(-L2*L3)/2.0;
	//z = sqrt(2*z);

	float c0 =  sqrt((L2-L1)/(L2-L3));
	float c0x = c0*V[2][V2];
	float c0y = c0*V[1][V2];
	float c0z = c0*V[2][V2];
	float c1 =  sqrt((L1-L3)/(L2-L3));
	float c1x = c1*V[0][V3];
	float c1y = c1*V[1][V3];
	float c1z = c1*V[2][V3];

	float z0 = -L3*c0x+L2*c1x;
	float z1 = -L3*c0y+L2*c1y;
	float z2 = -L3*c0z+L2*c1z;
	float s1,s2;
	s1=s2=1;
	float n0 = +s1*c0x+s2*c1x;
	float n1 = +s1*c0y+s2*c1y;
	float n2 = +s1*c0z+s2*c1z;

	//n0 = -L3*c0x-L2*c1x;
	//n1 = -L3*c0y-L2*c1y;
	//n2 = -L3*c0z-L2*c1z;
	
	//rotate the vector accordingly
	if (z2*z < 0){
		 z2 = -z2;
		 z1 = -z1;
		 z0 = -z0;
	//	 n0 = -n0;
	//	 n1 = -n1;
	//	 n2 = -n2;
	}
	result.x = z2*z;	
	result.y = -z0*z;	
	result.z = -z1*z;
	result.pitch = n0;//cos(segment.m1/segment.m0)/M_PI*180.0;
	result.roll = n1;//atan2(segment.v1,segment.v0)/M_PI*180.0;
	result.yaw = n2;//segment.v1/segment.v0;
	//result.roll = n2*z;	
	//result.pitch = -n0*z;	
	//result.yaw = -n1*z;
	return result;
}

STrackedObject CTransformation::transform(SSegment segment)
{
	float x,y,x1,x2,y1,y2,major,minor,v0,v1;
	STrackedObject result;
	//Transform to the Canonical camera coordinates
	x = segment.x;
	y = segment.y;
	transformXY(&x,&y);
	float m0 = segment.m0;
	float m1 = segment.m1;
	//if (segment.m0 > 0) m0 = sqrt(segment.m0);
	//if (segment.m1 > 0) m1 = sqrt(segment.m1);
	//major axis
	//vertices in image coords
	x1 = segment.x+segment.v0*m0*2;
	x2 = segment.x-segment.v0*m0*2;
	y1 = segment.y+segment.v1*m0*2;
	y2 = segment.y-segment.v1*m0*2;
	//printf("SSS: %.3f %.3f %.3f %.3f\n",segment.m0,segment.m1,segment.v0,segment.v1);
	//vertices in canonical camera coords 
	transformXY(&x1,&y1);
	transformXY(&x2,&y2);
//	printf("VVV: %.3f %.3f %.3f %.3f\n",x1,y1,x2,y2);
	//semiaxes length 
	major = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))/2.0;
	v0 = (x2-x1)/major/2.0;
	v1 = (y2-y1)/major/2.0;
	//printf("AAA: %f %f\n",sqrt((x-x1)*(x-x1)+(y-y1)*(y-y1))-sqrt((x-x2)*(x-x2)+(y-y2)*(y-y2)),major);

	//the minor axis 
	//vertices in image coords
	x1 = segment.x+segment.v1*m1*2;
	x2 = segment.x-segment.v1*m1*2;
	y1 = segment.y-segment.v0*m1*2;
	y2 = segment.y+segment.v0*m1*2;
	//vertices in canonical camera coords 
	transformXY(&x1,&y1);
	transformXY(&x2,&y2);
	//minor axis length 
	minor = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))/2.0;
	//printf("BBB: %f %f\n",sqrt((x-x1)*(x-x1)+(y-y1)*(y-y1))-sqrt((x-x2)*(x-x2)+(y-y2)*(y-y2)),minor);

	//Construct the ellipse characteristic equation
	float a,b,c,d,e,f;
	a = v0*v0/(major*major)+v1*v1/(minor*minor);
	b = v0*v1*(1.0/(major*major)-1.0/(minor*minor));
	c = v0*v0/(minor*minor)+v1*v1/(major*major);
	d = (-x*a-b*y);
	e = (-y*c-b*x);
	f = (a*x*x+c*y*y+2*b*x*y-1.0);
	
	//transformation to global coordinates
	double data[] ={a,b,d,b,c,e,d,e,f};

	result = eigen(data);

	return result;
}

