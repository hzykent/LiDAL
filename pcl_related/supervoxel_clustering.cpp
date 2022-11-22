#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/supervoxel_clustering.h>


// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

using namespace pcl;

int
main (int argc, char ** argv)
{
  if (argc < 2)
  {
    pcl::console::print_info ("Syntax is: %s {-p <pcd-file>} \n -o <output-file> \n-v <voxel resolution> \n-s <seed resolution> \n-c <color weight> \n-z <spatial weight> \n-n <normal_weight>] \n", argv[0]);
    return (1);
  }
  
  ///////////////////////////////  //////////////////////////////
  //////////////////////////////  //////////////////////////////
  ////// THIS IS ALL JUST INPUT HANDLING - Scroll down until 
  ////// pcl::SupervoxelClustering<pcl::PointXYZRGB> super
  //////////////////////////////  //////////////////////////////
  PointCloudT::Ptr cloud = boost::make_shared < PointCloudT >();
  bool pcd_file_specified = pcl::console::find_switch (argc, argv, "-p");
  std::string pcd_path;

  if (!pcd_file_specified)
  {
    std::cout << "No cloud specified!";
    return (1);
  }else
  {
    pcl::console::parse (argc,argv,"-p",pcd_path);
  }
  
  float voxel_resolution = 0.5f;
  bool voxel_res_specified = pcl::console::find_switch (argc, argv, "-v");
  if (voxel_res_specified)
    pcl::console::parse (argc, argv, "-v", voxel_resolution);
    
  float seed_resolution = 10.0f;
  bool seed_res_specified = pcl::console::find_switch (argc, argv, "-s");
  if (seed_res_specified)
    pcl::console::parse (argc, argv, "-s", seed_resolution);
  
  float color_importance = 0.1f;
  if (pcl::console::find_switch (argc, argv, "-c"))
    pcl::console::parse (argc, argv, "-c", color_importance);
  
  float spatial_importance = 0.4f;
  if (pcl::console::find_switch (argc, argv, "-z"))
    pcl::console::parse (argc, argv, "-z", spatial_importance);
  std::cout << "spatial_importance: " << spatial_importance << "\n";
  
  float normal_importance = 1.0f;
  if (pcl::console::find_switch (argc, argv, "-n"))
    pcl::console::parse (argc, argv, "-n", normal_importance);
  std::cout << "normal_importance: " << normal_importance << "\n";
  
  std::cout << "Loading pointcloud...\n";
  pcl::io::loadPCDFile (pcd_path, *cloud);
  std::cout << "Done making cloud!\n";

  ///////////////////////////////  //////////////////////////////
  //////////////////////////////  //////////////////////////////
  ////// This is how to use supervoxels
  //////////////////////////////  //////////////////////////////
  //////////////////////////////  //////////////////////////////
  
  pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution, false);
  super.setInputCloud (cloud);
  super.setColorImportance (color_importance);
  super.setSpatialImportance (spatial_importance);
  super.setNormalImportance (normal_importance);

  std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
 
  std::cout << "Extracting supervoxels!\n";
  super.extract (supervoxel_clusters);
  std::cout << "Found " << supervoxel_clusters.size () << " Supervoxels!\n";
  
  // Save
  std::string label_path;
  pcl::console::parse (argc,argv,"-o", label_path);
  PointLCloudT::Ptr full_labeled_cloud = super.getLabeledCloud ();
  pcl::io::savePCDFile (label_path, *full_labeled_cloud);
  return (0);
}


