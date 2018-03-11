cmake_minimum_required(VERSION 2.8.3)
project(mbzirc_mapping)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  std_srvs
  geometry_msgs
  object_detection
  message_generation
  dynamic_reconfigure
)


###############################################
## Declare ROS messages, services and actions ##
################################################
# Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   detecteddigit.msg
# )

## Generate services in the 'srv' folder
add_service_files(
  FILES
  GetObjects.srv
)

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
 )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )


###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES data_exporter
   CATKIN_DEPENDS roscpp std_msgs message_runtime
#  DEPENDS system_lib
   DEPENDS Eigen
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
#  include
)

## Declare a C++ library
#add_library(detector
#   src/detector.cpp
#)

## Declare a C++ executable
add_executable(mbzirc_mapping src/mbzirc_mapping.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(mbzirc_mapping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)

## Specify libraries to link a library or executable target against


target_link_libraries(mbzirc_mapping
   ${catkin_LIBRARIES}
)
