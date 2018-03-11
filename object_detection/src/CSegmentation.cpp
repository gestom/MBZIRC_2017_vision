#include "CSegmentation.h"

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

int compareSegments(const void* m1,const void* m2)
{
	if (((SSegment*)m1)->size >  ((SSegment*)m2)->size) return -1;
	if (((SSegment*)m1)->size <  ((SSegment*)m2)->size) return 1;
	return 0;
}

//cleanup and initialization
CSegmentation::CSegmentation()
{
	//cvtColor(*hsv,*hsv, COLOR_HSV2RGB);
	memset(colorArray,0,64*64*64);
	debug = true;
	drawSegments = true;

	float hta[] = {0, 50, 12,193, 15};
	float sta[] = {0, 54,202,142,212};
	float vta[] = {0,178,184,  6,219};
	memcpy(ht,hta,5*sizeof(float));
	memcpy(st,sta,5*sizeof(float));
	memcpy(vt,vta,5*sizeof(float));
}

CSegmentation::~CSegmentation()
{
}

//save the RGB grid to disk
void CSegmentation::saveColorMap(const char* name)
{
	FILE* f=fopen(name,"w");
	fwrite(colorArray,64*64*64,1,f); 
	fclose(f);
}

//load the RGB grid from disk
void CSegmentation::loadColorMap(const char* name)
{
	int stuff = 0;
	FILE* f=fopen(name,"r");
	if (f == NULL) return;
	stuff = fread(colorArray,64*64*64,1,f); 
	fclose(f);
}

//load color models 
int CSegmentation::loadColors(const char* name)
{
	FILE* file = fopen(name,"r");
	if (file == NULL) return -1;
	int index = 0;
	float h,s,v;
	for (int i=0;i<5;i++){
		fscanf(file,"%i %f %f %f\n",&index,&h,&s,&v);
		setColor(index,h,s,v);	
	}
	fclose(file);
	return 0;
}

//reset RGB index grid
void CSegmentation::resetColorMap()
{
	memset(colorArray,0,64*64*64);
}

//add pixel color to RGB index grid
void CSegmentation::learnPixel(Vec3b a,int type)
{
	int b = ((a[0]/4)*64+a[1]/4)*64+a[2]/4;
	colorArray[b] = type;
	int addr = 0;
	for (int r=0;r<256;r+=4){
		for (int g=0;g<256;g+=4){
			for (int b=0;b<256;b+=4){
				addr++;
			}
		}
	}
}

//add given pixel to the RGB grid
void CSegmentation::learnPixel(unsigned char* learned,int type)
{
	//convert it to HSV
	unsigned int learnedHue;
	unsigned char learnedSaturation;
	unsigned char learnedValue;
	rgbToHsv(learned[0],learned[1],learned[2],&learnedHue,&learnedSaturation,&learnedValue);

	//create a RGB grid
	unsigned char u[3];
	for (int r=0;r<256;r+=4){
		u[0] = r;
		for (int g=0;g<256;g+=4){
			u[1] = g;
			for (int b=0;b<256;b+=4){
				u[2] = b;
				int i = ((r/4)*64+g/4)*64+b/4;
				//colorArray[i] = evaluatePixel3(u);
				//if (classifyPixel(u) > 0) colorArray[i] = type;
			}
		}
	}
	fprintf(stdout,"Learned RGB: %i %i %i, HSV: %i %i %i\n",learned[0],learned[1],learned[2],learnedHue,learnedSaturation,learnedValue);
}

//returns a given pixel class, see Algorithms 3 and 4 of the paper
int CSegmentation::classifyPixel(Vec3b a)
{
	int b = ((a[0]/4)*64+a[1]/4)*64+a[2]/4;
	return colorArray[b];
}

SSegment CSegmentation::getSegment(int type,int number)
{
	SSegment result;
	result.x = segmentArray[number].x;
	result.y = segmentArray[number].y;
	return result;
}

//set pixels within HSV bounds as object ones
void CSegmentation::learnPixel(int minHue,int maxHue,int minSat,int maxSat,int minVal,int maxVal,int type)
{
	printf("%i %i %i %i %i %i\n", minHue,maxHue,minSat,maxSat,minVal,maxVal);
	Mat hsv = Mat(1, 1,CV_8UC3);
	Mat rgb = Mat(1, 1,CV_8UC3);
	if (minHue < 0) minHue = 0;
	if (minSat < 0) minSat = 0;
	if (minVal < 0) minVal = 0;
	if (maxHue > 179) maxHue = 179;
	if (maxSat > 255) maxSat = 255;
	if (maxVal > 255) maxVal = 255;

	/*go through the entire rgb index grid*/
	for (int r=0;r<64;r++){
		for (int g=0;g<64;g++){
			for (int b=0;b<64;b++){
				Vec3b v(r*4,g*4,b*4);
				rgb.at<Vec3b>(0,0) = v;
				cvtColor(rgb,hsv,COLOR_RGB2HSV);
				Vec3b a = hsv.at<Vec3b>(0,0);
				/*if a given cell is within hsv bounds, label it as type*/
				if (a(0) >= minHue && a(0) <= maxHue && a(1) >= minSat && a(1) <= maxSat && a(2) >= minVal && a(2) <=maxVal)
				{
					int i = (r*64+g)*64+b;
					colorArray[i] = type;
				}
			}
		}
	}


	/*go through the hsv bounds and add to the rgb grid*/
	for (int hue=minHue;hue<=maxHue;hue++){
		for (int sat=minSat;sat<=maxSat;sat++){
			for (int val=minVal;val<=maxVal;val++){
				Vec3b b( hue, sat, val );
				hsv.at<Vec3b>(0,0) = b;
				cvtColor(hsv,rgb, COLOR_HSV2RGB);
				Vec3b a = rgb.at<Vec3b>(0,0);
				int i = ((a[0]/4)*64+a[1]/4)*64+a[2]/4;
				colorArray[i] = type;
			}
		}
	}
}

SSegment CSegmentation::findSegment(Mat *image,Mat *coords,SSegment *output,int minSize,int maxSize)
{
	SSegment result;
	result.x = -1;
	result.y = -1;
	int width = image->cols;
	int height = image->rows;

	int expand[4] = {width,-width,1,-1};
	int* stack = (int*)calloc(width*height,sizeof(int));
	int* contour = (int*)calloc(width*height,sizeof(int));
	int stackPosition = 0;
	int contourPoints = 0;

	int type =0;
	numSegments = 0;
	int len = width*height;
	int *buffer = (int*)calloc(width*height,sizeof(int));

	/*'complete' mode (see paper) assumed -> label all image pixels accordingly*/
	for (int i = 0;i<len;i++) buffer[i] = -classifyPixel(image->at<Vec3b>(i/width,i%width));
	int borderType = 1000;

	/*label image borders to prevent flood fill running out of the image*/
	int topPos =  0;
	int bottomPos =  height-1;
	int leftPos =  0;
	int rightPos =  width-1;

	for (int i = leftPos;i<rightPos;i++){
		buffer[topPos*width+i] = borderType;	
		buffer[bottomPos*width+i] =  borderType;
	}

	for (int i = topPos;i<bottomPos;i++){
		buffer[width*i+leftPos] =  borderType;	
		buffer[width*i+rightPos] =  borderType;
	}

	int pos = 0;

	/** 4.2 search through the image **/ 
	int position = 0;
	int queueStart = 0;
	int queueEnd = 0;
	int nncount = 0;
	int dx,dy;
	int maxX,maxY,minX,minY;
	for (int i = 0;i<len;i++){

		/** 4.3 if pixel at position i has object color, initiate flood fill **/ 
		if (buffer[i] < 0 && numSegments < MAX_SEGMENTS){

			/***** Algorithm 3 starts here: flood fill segmentation (4.4) *****/

			/** initialize queue **/	
			queueStart = 0;
			queueEnd = 0;
			contourPoints = 0;
			segmentArray[numSegments].type = -buffer[i]; 

			//initialise segment data 
			segmentArray[numSegments].id = numSegments+1;
			segmentArray[numSegments].size = 1; 
			segmentArray[numSegments].warning = 0;

			/** initialise bounding box **/
			maxX = minX = segmentArray[numSegments].x = i%width; 
			maxY = minY = segmentArray[numSegments].y = i/width; 

			/** mark pixel as processed && increment segment ID **/
			type = buffer[i];
			buffer[i] = ++numSegments;

			/** push pixel position to the queue **/
			stack[queueEnd++] = i;

			/** #3.1 perform flood fill search **/ 
			while (queueEnd > queueStart){
				/** pull pixel from the queue **/
				position = stack[queueStart++];
				nncount = 0;
				/** #3.2 and check its neighbours **/ 
				for (int j =0;j<4;j++){
					pos = position + expand[j];

					int expand[4] = {width,-width,1,-1};
				
					/** #3.3 if not labeled, then label it **/
					//if (buffer[pos] == 0) buffer[pos] = -classifyPixel(image->at<Vec3b>(pos/width,pos%width));
					if (buffer[pos] != 0){
					
						nncount++;  //indicates the number of neighbouring pixes with object color -- not mentioned in the paper for clarity
					}
					

					/** #3.4 if it has potential object colour, then **/
					if (buffer[pos] == type)
					{
						/** add it to the queue **/
						stack[queueEnd++] = pos;
						/** and label it as belonging to the current segment **/
						buffer[pos] = numSegments;

						/** update bounding box info**/
						dx = pos%width;
						dy = pos/width;
						if (maxX < dx) maxX = dx;
						if (minX > dx) minX = dx;
						if (maxY < dy) maxY = dy;
						if (minY > dy) minY = dy;
					}
					// raise a warning flag if the object is touching the image border
					if (buffer[pos] == borderType) segmentArray[numSegments-1].warning = 1; 
				}
				//if it's a contour point, then add to the contour point array
				if (nncount != 4) contour[contourPoints++] = position;
			}
			statTotalSegments++;

			/**3.5 check segment size and roundness**/
			if (queueEnd > minSize){
				statGoodSizeSegments++;
			
				/**3.6 calculate segment roundness**/
				segmentArray[numSegments-1].roundness = fabs((maxX-minX+1)*(maxY-minY+1)*M_PI/4/queueEnd-1); 
				/**3.7 test segment roundness**/
				if (segmentArray[numSegments-1].roundness > minRoundness) queueEnd = 0; else statRoundSegments++;
			}else{
				queueEnd = 0;
			}
			/***** Algorithm 3 ends here ****/
			/***** Algorithm 4 takes over *****/
			if (queueEnd > 0){ 
				/**4.5 if the flood fill proposed a segment candidate**/

				//store contour points
				for (int s = 0;s<contourPoints;s++){
					pos = contour[s];
					buffer[pos] = 1000000+numSegments;	
				}

				/**4.6 calculate mean HSV, mean XY, and XY covariance**/
				long long int cx,cy,sx,sy,sr,sg,sb;
				long long int crr,crg,crb,cgg,cgb,cbb;
				long long int cxx,cxy,cyy; 
				int ch,cs,cv;
				maxX=maxY= -1;
				minX=minY = width*height;
				cxx=cxy=cyy=sx=sy=sr=sg=sb=0;

				for (int s = 0;s<queueEnd;s++){
					//retrive pixel position
					pos = stack[s];
					cx = pos%width; 
					cy = pos/width;
					//retrieve pixel color
					Vec3b vv = image->at<Vec3b>(cy,cx);
					//mean RGB calculation  
					sr += vv[0];
					sg += vv[1]; 
					sb += vv[2];

					//segment centroid calculation
					sx += cx;	
					sy += cy;
					//segment covariance 
					cxx += cx*cx; 
					cxy += cx*cy; 
					cyy += cy*cy; 
				}


				/**4.6a calculate mean HSV**/
				float fsr = (float)sr/queueEnd;
				float fsg = (float)sg/queueEnd;
				float fsb = (float)sb/queueEnd;

				//RGB to HSV conversion
				unsigned char ur = fsr; 
				unsigned char ug = fsg;  
				unsigned char ub = fsb;
				unsigned int uh;
				unsigned char us,uv;
				rgbToHsv(ur,ug,ub,&uh,&us,&uv);

				/**4.6b calculate segment centroid*/ 
				float fsx = (float)sx/queueEnd; 
				float fsy = (float)sy/queueEnd;

				/**4.6c calculate segment covariance (of x,y positions) */ 
				float fcxx = ((float)cxx/queueEnd-fsx*fsx);
				float fcxy = ((float)cxy/queueEnd-fsy*fsx);
				float fcyy = ((float)cyy/queueEnd-fsy*fsy);

				/**4.7 perform XY covariance eigenanalysis **/
				//determine eigenvalues by quadratic equation
				float det = (fcxx+fcyy)*(fcxx+fcyy)-4*(fcxx*fcyy-fcxy*fcxy);
				if (det > 0) det = sqrt(det); else det = 0;
				float eigvl0 = ((fcxx+fcyy)+det)/2;
				float eigvl1 = ((fcxx+fcyy)-det)/2;
				//calculate eigenvectors
				float eivec = (fcxy*fcxy+(fcxx-eigvl0)*(fcxx-eigvl0));
				if (fcyy != 0 && eivec > 0){                                                            
					segmentArray[numSegments-1].v0 = -fcxy/sqrt(eivec);
					segmentArray[numSegments-1].v1 = (fcxx-eigvl0)/sqrt(eivec);
				}else{
					segmentArray[numSegments-1].v0 = segmentArray[numSegments-1].v1 = 0;
					if (fcxx > fcyy) segmentArray[numSegments-1].v0 = 1.0; else segmentArray[numSegments-1].v1 = 1.0;
				}

				/**4.8 calculate segment circularity **/ 
				segmentArray[numSegments-1].circularity = M_PI*4*sqrt(eigvl1)*sqrt(eigvl0)/queueEnd; 		//note that because eigenvector length is 1, 2*sqrt(eigvl) equals to ellipse 
				printf("Circul: %f\n",segmentArray[numSegments-1].circularity);
				//segmentArray[numSegments-1].circularity = M_PI*4*eigvl1*eigvl0/queueEnd; 		//note that because eigenvector length is 1, 2*sqrt(eigvl) equals to ellipse 

				// store segment information 
				segmentArray[numSegments-1].m0 = sqrt(eigvl0); 
				segmentArray[numSegments-1].m1 = sqrt(eigvl1);
				segmentArray[numSegments-1].size = queueEnd; 
				segmentArray[numSegments-1].x = fsx;
				segmentArray[numSegments-1].y = fsy;

				// store segment mean HSV 
				segmentArray[numSegments-1].h = uh; 
				segmentArray[numSegments-1].s = us; 
				segmentArray[numSegments-1].v = uv;

				// store segment information 
				segmentArray[numSegments-1].minX = minX; 
				segmentArray[numSegments-1].minY = minY; 
				segmentArray[numSegments-1].maxX = maxX; 
				segmentArray[numSegments-1].maxY = maxY;

				//segmentArray[numSegments-1].roundness = eigvl1/eigvl0;

				// calculate corner candidates for the digit recognition module
				int corners = 0;
				int cX[4];
				int cY[4];
				float dist,maxDist;
				for (int cn = 0;cn<4;cn++){
					maxDist = 0;
					for (int s = 0;s<contourPoints;s++)
					{
						pos = contour[s];
						cx = pos%width-fsx; 
						cy = pos/width-fsy;
						dist = 0;
						if (cn > 0)
						{
							for (int c = 0;c<cn;c++) dist+=sqrt((cx-cX[c])*(cx-cX[c])+(cy-cY[c])*(cy-cY[c]));
						}else{
							dist = cx*cx+cy*cy;
							if (s < MAX_CONTOUR_POINTS){
								segmentArray[numSegments-1].contourX[s] = cx+fsx-minX;
								segmentArray[numSegments-1].contourY[s] = cy+fsy-minY;
							}
						}
						if (dist > maxDist)
						{
							cX[cn] = cx;	
							cY[cn] = cy;
							maxDist = dist;
						}
					}
				}
				segmentArray[numSegments-1].contourPoints = min(contourPoints,MAX_CONTOUR_POINTS);
				segmentArray[numSegments-1].combo = 1; 
				for (int ii = 0;ii<4;ii++){
					segmentArray[numSegments-1].cornerX[ii] = cX[ii]+fsx;
					segmentArray[numSegments-1].cornerY[ii] = cY[ii]+fsy;
				}

				//std::cout << "S "<<numSegments << "qEnd "<<queueEnd<<" fsx "<<fsx<<" cx "<<segmentArray[numSegments-1].x<<" fsy "<<fsy<<" cy "<<segmentArray[numSegments-1].y<< " seg size "<<(maxX-minX)<<","<<(maxY-minY)<< "combo " << segmentArray[numSegments-1].combo << std::endl;
				//if (peak > 0 && peak > queueEnd/100 && peak < queueEnd/10 && fabs(fsx-segmentArray[numSegments-1].x) < 5 && fabs(fsy-segmentArray[numSegments-1].y) < 5) segmentArray[numSegments-1].combo = 1; else segmentArray[numSegments-1].combo = 0;
				//segmentArray[numSegments-1].combo = peak;
				//if (segmentArray[numSegments-1].roundness >  1.0) segmentArray[numSegments-1].roundness  = 1.0/segmentArray[numSegments-1].roundness;
				/*real 'roundness' */
				//if (segmentArray[numSegments-1].roundness > minCircularity) segmentArray[numSegments-1].roundness = 4*sqrt(queueEnd)/contourPoints;
			}else{
				numSegments--;
			}
		}
	}
	printf("Through %i\n",numSegments);
	*coords = cv::Mat::ones(0, 1, CV_32FC2);
	Mat coord = cv::Mat::ones(1, 1, CV_32FC2);
	int taken = -1;

	/** 4.9 test the circularity of all segments **/
	for (int i = 0;i<numSegments;i++){
		if (fabs(segmentArray[i].circularity - 1)  < minCircularity && segmentArray[i].combo > 0 && segmentArray[i].warning == 0)
		{
			statCircularSegments++;
			if (taken == - 1) taken = segmentArray[i].id;
			coord.at<float>(0) = segmentArray[i].x;
			coord.at<float>(1) = segmentArray[i].y;
			segmentArray[i].type = -1;
			output[coords->rows] = segmentArray[i]; 
			coords->push_back(coord);
		}
	}

	//printf("Numsegments %i\n",realSegments);
	//vykreslime vysledek
	/*int j = 0;
	if (drawSegments){
		for (int i = 0;i<len;i++){
			j = buffer[i];
			if (j > 1000000 && segmentArray[j-1000000].roundness > minCircularity) image->at<Vec3b>(i/width,i%width) = Vec3f(0,0,0);
		}
	}*/
	free(buffer);
	free(stack);
	free(contour);
	return result;
}



void CSegmentation::setColor(int i,float h,float s,float v)
{
	if (i >=0 & i<5){ 	
		ht[i] = h;
		st[i] = s;
		vt[i] = v;
	}
}

int CSegmentation::classifySegment(SSegment s)
{
	int mindex = s.type;
	if (mindex == -1){
		float minimal = 1000000;
		for (int i = 1;i<5;i++){
			float dh = ht[i]-s.h; 
			float ds = st[i]-s.s; 
			float dv = vt[i]-s.v; 
			float mina = sqrt(dh*dh);//(+ds*ds+dv*dv);
			//printf("Clas: %.3f  %.3f %.3f %.3f %.3f %.3f %.3f\n",s.h,s.s,s.v,ht[i],st[i],vt[i],mina);
			if (mina <minimal)
			{
				minimal=mina;
				mindex=i;
			}
		}
	}
	if (mindex == 4)mindex = 5;
	return mindex;
}

//prevod RGB -> HSV, prevzato z www
void CSegmentation::rgbToHsv(unsigned char r, unsigned char  g, unsigned char b, unsigned int *hue, unsigned char *saturation, unsigned char *value )
{
	float min, max, delta;
	float h,s,v;   

	h=s=v=0; 
	*saturation = (unsigned char) s;
	*value = (unsigned char) v;
	*hue = (unsigned int) h;

	min = min( r, min(g, b) );
	max = max( r, max(g, b) );
	v = max;			

	delta = max - min;

	if( max != 0 )
		s = min(delta*255 / max,255);	
	else {
		s = 0;
		h = -1;
		return;
	}

	if( r == max )
		h = ( g - b ) / delta;		// between yellow & magenta
	else if( g == max )
		h = 2 + ( b - r ) / delta;	// between cyan & yellow
	else
		h = 4 + ( r - g ) / delta;	// between magenta & cyan
	h = h*60;
	if (h<0) h+=360;
	*saturation = (unsigned char) s;
	*value = (unsigned char) v;
	*hue = (unsigned int) h;
}

