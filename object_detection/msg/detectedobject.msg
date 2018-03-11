int32 id		#equal to the published image header::seq
time timestamp		#equal to the published image header::time - camera delay
int32 imageID		#equal to the input image sequence
int32 type		#segment type
float32 x		#centroid
float32 y		#centroid
float32 h		#color - Hue
float32 s		#color - Saturation 
float32 v		#color - Value 
int32 minX		#bounding box data
int32 minY		#bounding box data
int32 maxX		#bounding box data
int32 maxY		#bounding box data
int32 size 		#pixel area
geometry_msgs/Pose 	relative 
geometry_msgs/Pose 	absolute 
geometry_msgs/Pose 	visual 
geometry_msgs/Pose 	uncorrected 
int32[] cornerX		#corners 
int32[] cornerY		#corners
int32[] contourX	#contour 
int32[] contourY	#contour
float32 roundness
float32 circularity
bool full		#fully in the field of view

float32 px		#metric positions X
float32 py		#metric positions Y
float32 pz		#metric positions Z
float32 d		#distance

#	float ch,cs,cv
