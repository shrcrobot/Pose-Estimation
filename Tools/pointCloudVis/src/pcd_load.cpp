#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>//which contains the required definitions to load and store point clouds to PCD and other file formats.
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

main (int argc, char **argv)
{
	float radiusSearch;
	int cntNei;
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
	    ("help", "produce help message")
	    ("-r", po::value<float>(), "setRadiusSearch")
	    ("-n", po::value<int>(), "setMinNeighborsInRadius")
	    ("-p", po::value<bool>(), "switch of passthrough")
	    ("-s", po::value<bool>(), "switch of ransac")
	    ("-o", po::value<bool>(), "switch of RadiusOutlierRemoval")
	    ("-b", po::value<bool>(), "switch of boundaryRemoval")
	    ("-c", po::value<bool>(), "switch of Clustering")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    

	if (vm.count("help")) {
	    std::cout << desc << "\n";
	    return 1;
	}

	if (vm.count("-r")) {
		radiusSearch=vm["-r"].as<float>();
	    std::cout << "setRadiusSearch was set to " 
	 << vm["-r"].as<float>() << ".\n";
	} else {
		radiusSearch=0.0136;
	    std::cout << "setRadiusSearch was set to default(0.0136).\n";
	}

	if (vm.count("-n")) {
		cntNei=vm["-n"].as<int>();
	    std::cout << "RadiusSearch was set to " 
	 << vm["-n"].as<int>() << ".\n";
	} else {
		cntNei=126;
	    std::cout << "MinNeighborsInRadius was set to default(126).\n";
	}

	bool s_ps=true;
	if (vm.count("-p")) {
		s_ps=vm["-p"].as<bool>();
	    std::cout << "PassThrough is "<< vm["-p"].as<bool>() <<std::endl;
	}

	bool s_ransac=true;
	if (vm.count("-s")) {
		s_ransac=vm["-s"].as<bool>();
	    std::cout << "ransac is "<< vm["-s"].as<bool>()  <<std::endl;
	}

	bool s_ror=true;
	if (vm.count("-o")) {
		s_ror=vm["-o"].as<bool>();
	    std::cout << "RadiusOutlierRemoval is " << vm["-o"].as<bool>() <<std::endl;
	}

	bool s_boundary=true;
	if (vm.count("-b")) {
		s_boundary=vm["-b"].as<bool>();
	    std::cout << "BoundaryRemoval is " << vm["-b"].as<bool>() <<std::endl;
	}

	bool s_cluster=true;
	if (vm.count("-c")) {
		s_cluster=vm["-c"].as<bool>();
	    std::cout << "Clustering is" << vm["-c"].as<bool>() <<std::endl;
	}

  ros::init (argc, argv, "UandBdetect");
  ros::NodeHandle nh;
  ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("pcl_output", 1);

  //seg
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);


  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud=(pcl::PointCloud<pcl::PointXYZRGB>::Ptr)&cloud;
  sensor_msgs::PointCloud2 output;
  pcl::io::loadPCDFile ("/home/shrc/box_plane.pcd", cloud);
  //Convert the cloud to ROS message


  if(s_ps){
	  pcl::PassThrough<pcl::PointXYZRGB> Cloud_Filter; // Create the filtering object
	  Cloud_Filter.setInputCloud (pCloud);           // Input generated cloud to filter
	  Cloud_Filter.setFilterFieldName ("z");        // Set field name to Z-coordinate
	  Cloud_Filter.setFilterLimits (0.4, 0.7);      // Set accepted interval values
	  Cloud_Filter.filter (*pCloud);              // Filtered Cloud Outputted
  }
 

 if(s_ransac){
 	  seg.setModelType (pcl::SACMODEL_PLANE);
	  seg.setMethodType (pcl::SAC_RANSAC);
	  seg.setDistanceThreshold (0.01);
	  seg.setInputCloud (pCloud);
	  seg.segment (*inliers, *coefficients);

	  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	  extract.setInputCloud(pCloud);
	  extract.setIndices(inliers);
	  extract.setNegative(true);
	  extract.filter(*pCloud);
 }
/*
  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    return (-1);
  }

  std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;


  std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
  for (size_t i = 0; i < inliers->indices.size (); ++i)
    std::cerr << inliers->indices[i] << "    " << newCloud->points[inliers->indices[i]].x << " "
                                               << newCloud->points[inliers->indices[i]].y << " "
                                               << newCloud->points[inliers->indices[i]].z << std::endl;
*/

 if(s_ror){
 	  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> rorfilter (true); // Initializing with true will allow us to extract the removed indices
	  rorfilter.setInputCloud (pCloud);
	  rorfilter.setRadiusSearch (radiusSearch);
	  rorfilter.setMinNeighborsInRadius (cntNei);
	  rorfilter.setNegative (false);
	  rorfilter.filter (*pCloud);
	  // The resulting cloud_out contains all points of cloud_in that have 4 or less neighbors within the 0.1 search radius
	  //const pcl::IndicesConstPtr & indices_rem = rorfilter.getRemovedIndices();
	  // The indices_rem array indexes all points of cloud_in that have 5 or more neighbors within the 0.1 search radius
	  //std::cerr<< indices_rem->size() << std::endl;
 }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr remainders(new pcl::PointCloud<pcl::PointXYZRGB>);
  if(s_boundary){
	// compute normals; 
	pcl::search::Search<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>()); 
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>); 
	pcl::NormalEstimation<pcl::PointXYZRGB,pcl::Normal> normal_est; 
	normal_est.setSearchMethod(tree); 
	normal_est.setInputCloud(pCloud); 
	normal_est.setKSearch(50); 
	normal_est.compute(*normals); 
	//normal_est.setViewPoint(0,0,0); 


	//calculate boundary; 
	pcl::PointCloud<pcl::Boundary> boundary; 
	pcl::BoundaryEstimation<pcl::PointXYZRGB,pcl::Normal,pcl::Boundary> boundary_est; 
	boundary_est.setInputCloud(pCloud); 
	boundary_est.setInputNormals(normals); 
	boundary_est.setRadiusSearch(0.02); 
	//boundary_est.setAngleThreshold(PI/4); 
	boundary_est.setSearchMethod(pcl::search::KdTree<pcl::PointXYZRGB>::Ptr(new pcl::search::KdTree<pcl::PointXYZRGB>)); 
	boundary_est.compute(boundary); 

	int cnt=0;
    for (int i=0; i<pCloud->size(); i++){
		uint8_t x = (boundary.points[i].boundary_point);
	    int a = static_cast<int>(x);
	    if ( a != 1)
	    {
			( *remainders).push_back(pCloud->points[i]);
			cnt++;
		}
    }


/*  visualization    
	std::cout<<"boudary size is：" <<cnt <<std::endl;
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));
	viewer->addPointCloud<pcl::PointXYZRGB>(pCloud,"cloud");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(pCloud,normals,20,0.03,"normals");
	while(!viewer->wasStopped()){
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	//pcl::visualization::CloudViewer viewer ("test");
    //viewer.showCloud(boundPoints);
*/
  }else{
      remainders=pCloud;
  }



  pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusteredRemainder(new pcl::PointCloud<pcl::PointXYZRGB>);
  if(s_cluster){
  	// https://blog.csdn.net/weixin_41038905/article/details/80976948
  	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud (remainders);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance (0.02); //设置近邻搜索的搜索半径为2cm
	ec.setMinClusterSize (5000);
	ec.setMaxClusterSize (100000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (remainders);
	ec.extract (cluster_indices);


	int cnt=0;

	struct COLOR
	{
		int r;
		int g;
		int b;
		COLOR(int red,int green,int blue){
			r=red;
			g=green;
			b=blue;
		}
	};

	COLOR red(220,20,60);
	COLOR green(0,128,0);
	COLOR yellow(255,255,0);
	COLOR blue(30,144,255);
	COLOR white(255,255,255);


	std::vector<COLOR> colors;
	colors.push_back(red);
	colors.push_back(green);
	colors.push_back(yellow);
	colors.push_back(blue);
	colors.push_back(white);

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin ();
	 it != cluster_indices.end(); ++it)
	{
		COLOR c=colors[cnt%5];
		//std::cout<<c.r<<" "<<c.g<<" "<<c.b<<std::endl;
		uint32_t rgb=((uint32_t)c.r<<16 | (uint32_t)c.g<<8 | (uint32_t)c.b);
	    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
	    for (std::vector<int>::const_iterator pit = it->indices.begin();
	     pit != it->indices.end (); ++pit)
	    {
	        //cloud_cluster->points.push_back (remainders->points[*pit]);
	        //cloud_cluster->width = cloud_cluster->points.size ();
	        //cloud_cluster->height = 1;
	        //cloud_cluster->is_dense = true; 
	    	auto* cpt = &(remainders->points[*pit]);
	    	cpt->rgb=*reinterpret_cast<float*>(&rgb);
	    	(*clusteredRemainder).push_back(remainders->points[*pit]);
	    }
	    // pcl::visualization::CloudViewer viewer("Cloud Viewer");
	    // viewer.showCloud(cloud_cluster);
	    // pause();
	    cnt++;
	}
  }


  pcl::toROSMsg(*clusteredRemainder, output);

  std::cout<<"Outputted!"<<std::endl;
  output.header.frame_id = "odom";//this has been done in order to be able to visualize our PointCloud2 message on the RViz visualizer
  ros::Rate loop_rate(1);
  while (ros::ok())
  {
    pcl_pub.publish(output);
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}