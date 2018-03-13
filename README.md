# MBZIRC 2017 CTU-UPENN-UoL team Vision Systems

This repo contains the vision systems and relevant datasets used by the joint team of [Multirobot Systems Group](https://mrs.felk.cvut.cz) of the [Czech Technical University](https://www.cvut.cz) (CZ), [GRASP](https://grasp.upenn.edu) at [University of Pennsylvania](https://www.upenn.edu) (USA), and [Lincoln Centre for Autonomous Systems Research](http://lcas.lincoln.ac.uk) at [University of Lincoln](https://lincoln.ac.uk) (UK) during the Mohammed bin Zayed International Robotics Challenge [MBZIRC](https://Www.mbzirc.com).

These vision system were used in the MBZIRC Challenge I (automated landing) and MBZIRC Challenge III (cooperative object collection).
Both vision systems are available here as [ROS](https://www.ros.org) nodes with associated config and launch files.
Their description is provided in a submission for Journal of Field Robotics, (ROB-17-0117), which is currently in review.

Note that the original system was running with Ubuntu version 14.04 and ROS indigo.
For the sake of compatibility, the codes provided here were updated for compatibility with ROS kinetic and Ubuntu 16.04.
Moreover, we performed an extensive cleanup, removal of ballast and legacy code, and added comments, so that one can easily relate the code blocks to the algorithm description of the aforementioned article.
To verify if the aforementioned changes affected the systems' performance, we re-run the experiments described in Section 4 of the article and obtained similar results as in the paper.

## Automated landing 

### Code 

The landing pattern detection is done by *cross_detector* node [cross_detector.cpp](https://github.com/gestom/MBZIRC_2017_vision/blob/master/cross_detector/src/cross_detector.cpp).
The core of vision detection is in the [detector.cpp](https://github.com/gestom/MBZIRC_2017_vision/blob/master/cross_detector/src/detector.cpp).
The codes contains comments with respect to Section 2 of the submitted paper.

To see debug information you can uncomment some definition inside the detector.cpp file.

To quickly test the code, build the node using *catkin_make*, download one of the [rosbags](https://drive.google.com/open?id=12JMtMRwAxuQDOIvuAB3Pw12v5sdFLBfq), then run the node using the *test.launch*:

```roslaunch cross_detector test.launch```

If everything goes well, you should see the pattern detection output similar to this video:
[![Landing pattern detection example](https://github.com/gestom/MBZIRC_2017_vision/blob/master/landing.jpg)](https://youtu.be/rBRpL17b66s)

### Datasets

The datasets collected during the contest are accessible on google drive folder [MBZIRC2017:  Landing datasets](https://drive.google.com/drive/folders/1Er9TcR6by7SloJWd5UItLLAvSvsILtbK?usp=sharing)

## Treasure hunt (colored object collection)

This vision system is intended for the `Treasure hunt' scenario, where several UAVs search for small objects in the contest arena, pick them up and deliver to a specific location.

### Code 

The detection and localisation is performed by *mbzirc_detector* node, which is written in cpp in [detector.cpp](https://github.com/gestom/MBZIRC_2017_vision/blob/master/object_detection/src/detection.cpp).
The core of the detection is in the [CSegmentation](https://github.com/gestom/MBZIRC_2017_vision/blob/master/object_detection/src/CSegmentation.cpp) class and the core of the 3d localisation is in the [CTransformation](https://github.com/gestom/MBZIRC_2017_vision/blob/master/object_detection/src/CTransformation.cpp) class.
The mapping method runs in a separate package. 
These codes contain comments numbered in the same way as in Algorithm 3,4,5 of the article, so one can relate the blocks of actual cpp code to the code of the paper.  

To quickly test the code, build the nodes using *catkin_make*, download one of the [rosbags](https://drive.google.com/open?id=12JMtMRwAxuQDOIvuAB3Pw12v5sdFLBfq), then run the node using the *rosbag.launch*:

```UAV_NAME=uav2 roslaunch object_detection rosbag.launch```

Once the node displays three empty windows, play the downloaded rosbag:
 
```rosbag play Grand_Challenge_flight_1_UAV_2.bag```

If everything goes well, one of the windows should show the view from the UAV, which should be similar to the video:

[![Object detection in the treasure hunt scenario](https://github.com/gestom/MBZIRC_2017_vision/blob/master/treasure.jpg)](https://youtu.be/mpUrTWHK3N8)

This video is related to the Experiment described in Section 4.2 of the paper.

### Datasets
 
The datasets collected during the contest are accessible on google drive folder [MBZIRC2017:  Treasure hunt vision datasets](https://drive.google.com/open?id=1tWgxXvr7SaWj2Dd4iZk2PaG5IJbCLQMM).

## Dependencies

Please make sure that the following dependencies are installed:

```sudo apt-get install ros-kinetic-image-transport-plugins```
