
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
 
unsigned int pt_num;

int main(int argc, char** argv){
    MyIO my_io;
    string h5_filename;
    string class_name;
    if(argc>=2){
        h5_filename = string(argv[1])+".h5";
    }else{
        h5_filename = "dataset.h5";
    }
    if(argc>=3){
        pt_num = atoi(argv[2]);
    }else{
        pt_num = 1024;
    }
    if(argc>=4){
        class_name = string(argv[3]);
    }else{
        class_name = "raw_data";
    }
 
    my_io.combinePCDsAndLabelsIntoH5File(h5_filename, class_name, "./"+class_name+"/pcd_names_file.txt", "./"+class_name+"/labels_file.txt", "./"+class_name+"/angles_file.txt", "./"+class_name+"/centers_file.txt");
 
    cout << "data is writed to " << h5_filename << endl;
    return 0;
}
