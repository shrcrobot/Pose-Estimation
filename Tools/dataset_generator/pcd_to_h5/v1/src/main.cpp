
/*Author: AmbitiousRuralDog
Date: 2018/06/28
This a main file to call my_io functions to covert PCL's PCD-type
data to chunk-format HDF5 data
*/
 
#include <iostream>
#include <string>
#include <time.h>
#include <fstream>
#include "../include/my_io.h"
 
using namespace std;
 
int main(int argc, char** argv){
    MyIO my_io;
    string h5_filename = "trainset1.h5";
 
    my_io.combinePCDsAndLabelsIntoH5File(h5_filename, "./raw_data/pcd_names_file.txt", "./raw_data/labels_file.txt");
 
    cout << "data is writed to " << h5_filename << endl;
    return 0;
}
