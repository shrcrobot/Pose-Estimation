
/*Author: AmbitiousRuralDog
Date: 2018/06/28
This a header file to declare my_io's functions that covert PCL's PCD-type
data to chunk-format HDF5 data
*/
 
#ifndef MYIO_H
#define MYIO_H
#include "H5Cpp.h"
 
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
 
#include <iostream>
#include <string>
#include <cstdio>
#include <fstream>
 
using namespace std;
 
typedef pcl::PointXYZI PT;
typedef pcl::PointCloud<PT> PCT;
 
class MyIO
{
public:
  MyIO();
  ~MyIO();

  int markDownStoredPCDNameAndItsLabel(const string &pcd_name, const int &label, const string &pcd_names_file, const string &labels_file);
  int combinePCDsAndLabelsIntoH5File(const string &h5_file, const string &pcd_names_file, const string &labels_file);
  int readFileAndCountHowManyClouds(const string &pcd_names_file);

private:
  int readPCDs(const string &pcd_names_file, float *data, const unsigned int &pt_num);
  int readLabels(const string &labels_file, int *data);
};
 
#endif // MYIO_H
