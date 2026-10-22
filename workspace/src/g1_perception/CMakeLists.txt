cmake_minimum_required(VERSION 3.8)
project(g1_perception)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # The tracker is arithmetic in tight loops; an unoptimised build is ~10x slower and misses
  # the camera rate by a wide margin. colcon's default build type is empty.
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io features filters kdtree search registration sample_consensus)

# The tracking core: no ROS, so the offline bench and the tests drive exactly what the node runs.
#
# PCL_DEFINITIONS is deliberately NOT applied. On some distributions it carries -march=native,
# and Eigen objects laid out under one instruction set and used under another misalign: the
# classic symptom is a crash inside Eigen in code that is plainly correct. PCL's headers need
# only its include directories.
add_library(g1_perception_core
  src/object_model.cpp
  src/icp_tracker.cpp
  src/pose_spec.cpp
  src/multi_tracker.cpp
  src/debug_render.cpp
)
target_include_directories(g1_perception_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_compile_features(g1_perception_core PUBLIC cxx_std_17)
target_link_libraries(g1_perception_core PUBLIC Eigen3::Eigen ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_library(g1_object_tracker_node src/object_tracker_node.cpp)
target_link_libraries(g1_object_tracker_node PUBLIC g1_perception_core)
ament_target_dependencies(g1_object_tracker_node PUBLIC
  rclcpp ament_index_cpp sensor_msgs geometry_msgs vision_msgs diagnostic_msgs visualization_msgs tf2 tf2_ros)

add_executable(g1_object_tracker src/object_tracker_main.cpp)
target_link_libraries(g1_object_tracker g1_object_tracker_node)

# The head depth camera as a sparse cloud for MoveIt's octomap (see the file's header).
add_executable(g1_depth_cloud src/depth_cloud_main.cpp)
ament_target_dependencies(g1_depth_cloud rclcpp sensor_msgs)

# Offline: replays a recorded or rendered sequence and grades it against ground truth.
add_executable(icp_bench tools/icp_bench.cpp)
target_link_libraries(icp_bench g1_perception_core)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS g1_perception_core g1_object_tracker_node
  EXPORT export_${PROJECT_NAME} DESTINATION lib)
install(TARGETS g1_object_tracker g1_depth_cloud icp_bench DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS tools/make_icp_dataset.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch models rviz DESTINATION share/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(Eigen3 PCL)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_tracker test/test_tracker.cpp)
  target_link_libraries(test_tracker g1_perception_core)
endif()

ament_package()
