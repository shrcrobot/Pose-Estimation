
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

extern unsigned int pt_num;
typedef pcl::PointXYZI PT;
typedef pcl::PointCloud<PT> PCT;

class MyIO
{
public:
  MyIO();
  ~MyIO();

  int markDownStoredPCDNameAndItsLabel(const string &pcd_name, const int &label, const string &pcd_names_file, const string &labels_file);
  int combinePCDsAndLabelsIntoH5File(const string &h5_file, string class_name, const string &pcd_names_file, const string &labels_file, const string &angles_file, const string &centers_file);
  int readFileAndCountHowManyClouds(const string &pcd_names_file);

private:
  int readPCDs(const string &pcd_names_file, string class_name, float *data, const unsigned int &pt_num);
  int readLabels(const string &labels_file, int *data);
  int readAngles(const string &angles_file, float *data);
  int readCenters(const string &centers_file, float *data);
  void Split(const string& src, const string& separator, float* array);

};
 
#endif // MYIO_H
