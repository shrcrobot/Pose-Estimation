/*Author: AmbitiousRuralDog
Date: 2018/06/28
This a cpp file to define my_io's functions that covert PCL's PCD-type
data to chunk-format HDF5 data
*/
#include "../include/my_io.h"
 
MyIO::MyIO()
{
 
}
 
MyIO::~MyIO(){
 
}
// This function is to store the label and corresponding pcd_file name into two seperate txt files.
// These txt files aim to be plugged in PointNet for training
int MyIO::markDownStoredPCDNameAndItsLabel(const string &pcd_name, const int &label, const string &pcd_names_file, const string &labels_file){
    ofstream outfile, outfile2;
    outfile.open(pcd_names_file, ofstream::app);
    outfile2.open(labels_file, ofstream::app);
    if (!outfile.is_open() | !outfile2.is_open())
        return -1;
    outfile << pcd_name + "\n";
    outfile2 << to_string(label) + "\n";
    outfile.close();
    outfile2.close();
 
    return 1;
}
 
// Extract a set of PCD files and combine them and write the result to a h5 file.
// Also write lables to a h5 file.
int MyIO::combinePCDsAndLabelsIntoH5File(const string &h5_file, const string &pcd_names_file, const string &labels_file){
    unsigned int RANK_clouds = 3;
    unsigned int RANK_labels = 2;
 
    unsigned int pt_dim = 3;
    unsigned int pt_num = 1024;
    // From a file storing all PCD filenames, dynamically count how many PCD files are required to put into H5 file
    unsigned int cloud_num;
    cloud_num = readFileAndCountHowManyClouds(pcd_names_file);
    if (cloud_num == -1) return 0;
 
    const std::string DATASET_NAME("data");
    const std::string LABELSET_NAME("label");
    // Read clouds and labels and store as float array and int array respectively
    float* data = new float [pt_dim*pt_num*cloud_num];
    readPCDs(pcd_names_file, data, pt_num);
    int* label = new int[cloud_num];
    readLabels(labels_file, label);
 
    // What is the size of each chunk of data
    unsigned int cloud_chunksize = unsigned(floor(cloud_num/8));
    if (cloud_chunksize < 1) cloud_chunksize = 1;
    unsigned int pt_num_chunksize = unsigned(floor(pt_num/8));
    if (pt_num_chunksize < 1) pt_num_chunksize = 1;
 
    try
    {
        hid_t file_id;
        // Open a h5 file for clouds
        file_id = H5Fcreate(h5_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
 
        hid_t space_id, dataset_id, chunk_dataset_id;
        // Create a dataset
        dataset_id = H5Pcreate(H5P_DATASET_CREATE);
        // Create a dataspace for dataset
        hsize_t dims[RANK_clouds] = {cloud_num, pt_num, pt_dim};
        hsize_t dims_max[RANK_clouds] = {cloud_num, pt_num, pt_dim};
        hsize_t chunk_dims[RANK_clouds] = {cloud_chunksize, pt_num_chunksize, 1};
        H5Pset_chunk(dataset_id, RANK_clouds, chunk_dims);
        space_id = H5Screate_simple(RANK_clouds, dims, dims_max);
        // Change the dataset into a chunk-format dataset
        chunk_dataset_id = H5Dcreate1(file_id,"data",H5T_NATIVE_FLOAT,space_id,dataset_id);
        // Write data into chunk-format dataset
        H5Dwrite(chunk_dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        // Close dataspace dataset
        H5Dclose(chunk_dataset_id);
        H5Sclose(space_id);
        H5Pclose(dataset_id);
 
        hid_t space_id2, dataset_id2, chunk_dataset_id2;
        // Create a dataset
        dataset_id2 = H5Pcreate(H5P_DATASET_CREATE);
        // Create a dataspace for dataset
        hsize_t dims2[RANK_labels] = {cloud_num, 1};
        hsize_t dims2_max[RANK_labels] = {cloud_num,1};
        hsize_t chunk_dims2[RANK_labels] = {cloud_num, 1};
        H5Pset_chunk(dataset_id2, RANK_labels, chunk_dims2);
        space_id2 = H5Screate_simple(RANK_labels, dims2, dims2_max);
        // Change the dataset into a chunk-format dataset
        chunk_dataset_id2 = H5Dcreate1(file_id,"label",H5T_NATIVE_INT,space_id2,dataset_id2);
        // Write data into chunk-format dataset
        H5Dwrite(chunk_dataset_id2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, label);
        // Close dataspace dataset and h5 file
        H5Dclose(chunk_dataset_id2);
        H5Sclose(space_id2);
        H5Pclose(dataset_id2);
        H5Fclose(file_id);
    }
 
    // catch failure caused by the H5File operations
    catch(H5::FileIException error)
    {
        error.printError();
        return -1;
    }
    // catch failure caused by the DataSpace operations
    catch(H5::DataSpaceIException error)
    {
        error.printError();
        return -1;
    }
    // catch failure caused by the Group operations
    catch(H5::GroupIException error)
    {
        error.printError();
        return -1;
    }
    // catch failure caused by the DataSet operations
    catch(H5::DataSetIException error)
    {
        error.printError();
        return -1;
    }
    return 1;
}
 
// From a file storing all PCD filenames, dynamically count how many PCD files are required to put into H5 file
int MyIO::readFileAndCountHowManyClouds(const string &pcd_names_file){
    ifstream in;
    in.open(pcd_names_file);
    if (!in.is_open())
        return -1;
    string textline;
    unsigned int count = 0;
    while (getline(in, textline)){
        count++;
    }
    in.close();
    return count;
}
 
// From a file storing all PCD filenames, combine all cloud data to a float array
int MyIO::readPCDs(const string &pcd_names_file, float *cloud_array, const unsigned int &pt_num){
    ifstream in;
    in.open(pcd_names_file);
    if (!in.is_open())
        return -1;
 
    string textline;
    unsigned int idx = 0;
 
    while (getline(in, textline)){
        PCT::Ptr a_cloud (new PCT);
        pcl::io::loadPCDFile("./raw_data/"+textline,*a_cloud);
        cout<<textline<<endl;
        if (pt_num!=a_cloud->points.size())
        {
            cout << "Error: A cloud's point number is not equal " << pt_num << endl;
        }
        for (int j = 0; j < pt_num; j++){
            cloud_array[idx] = a_cloud->points[j].x;
            cloud_array[idx+1] = a_cloud->points[j].y;
            cloud_array[idx+2] = a_cloud->points[j].z;
            idx = idx+3;
        }
    }
    return 1;
}
// From a file storing all PCD labels, combine all labels to a int array
int MyIO::readLabels(const string &labels_file, int *label_array){
    ifstream in;
    in.open(labels_file);
    if (!in.is_open())
        return -1;
 
    unsigned int count = 0;
    int a_label;
 
    while (in >> a_label){
        label_array[count] = a_label;
        count++;
    }
    return 1;
}