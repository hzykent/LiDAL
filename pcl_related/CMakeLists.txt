cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(supervoxel_clustering)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (supervoxel_clustering supervoxel_clustering.cpp)
target_link_libraries (supervoxel_clustering ${PCL_LIBRARIES})
