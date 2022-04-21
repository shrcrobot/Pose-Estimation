# Object Pose Estimation with Point Cloud Data for Robot Grasping


#### Codes

Env: Python 3.6 + PyTorch 1.0.1

classifier.py: Classification network structure

rotation_cls_new.py: Posture estimation network structure

center_est_pt.py: Geometric center estimation network structure

train_*: training codes for networks

eval_*: evaluation codes for networks



#### Tools

1.blensor - Python script for blender with blensor:  Generating object's unilateral point cloud image from mesh models.

2.labels_generator - generating pcd_names_file.txt & labels_file.txt for pcd file.

3.dataset_generator:

 - pcd_prep: upsample and downsample pcd point cloud images to fixed point number.
 - pcd_to_h5: Packing pcd and labels to hdf5 format dataset.

4.pcl-color: capturing point cloud scene from Realsense depth camera.

5.pointCloudVis: Preprocessing scene and display point cloud images in rviz (a 3D visualization tool for ROS).



#### Dataset

Different versions of the objects' unilateral point cloud dataset.

#### Pre-trained Models

Pre-trained models for each network.

#### Acknowledgment

The point cloud images used in this work are rendered from mesh models in KIT object models database and 3DNet database.

[1] Alexander Kasper, Zhixing Xue and Rüdiger Dillmann. “The KIT object models database: An objectmodel database for object recognition, localization and manipulation in service robotics”. TheInternational Journal of Robotics Research, 2012, 31(8): 927–934.

[2] Walter Wohlkinger, Aitor Aldoma, Radu B Rusu et al. “3dnet: Large-scale object class recognitionfrom cad models”. In: 2012 IEEE international conference on robotics and automation, 2012: 5384–5391.
