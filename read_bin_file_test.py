# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:25:28 2020

@author: HarryShen
"""    
import numpy as np
#import mayavi.mlab
#import utils.kitti_aug_utils as augUtils
#import utils.kitti_bev_utils as bev_utils
import utils.config as cnf
import cv2, torch
import pyzed.sl as sl
import pcl.pcl_visualization
from FastSCNN.FastSCNN import FastSCNN

#'''
Raw_image = cv2.imread("./sampledata/image_2/000000.png")
visual = pcl.pcl_visualization.CloudViewing()
pointcloud = np.fromfile(str('./sampledata/velodyne/000000.bin'), dtype=np.float32).reshape(-1,4)
print(pointcloud)
x = pointcloud[:, 0]  # x position of point
y = pointcloud[:, 1]  # y position of point
z = pointcloud[:, 2]  # z position of point
r = pointcloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
 
vals='height'
if vals == "height":
    col = z
else:
    col = d
 
#fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
#mayavi.mlab.points3d(x, y, z,
#                     col,          # Values used for Color
#                     mode="point",
#                     colormap='spectral', # 'bone', 'copper', 'gnuplot'
#                     # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
#                     figure=fig,
#                     )
# 
#x_ = np.linspace(5,5,50)
#y_ = np.linspace(0,0,50)
#z_ = np.linspace(0,5,50)
#mayavi.mlab.plot3d(x_, y_, z_)
#mayavi.mlab.show()


#color_cloud = pcl.PointCloud_PointXYZRGBA(pointcloud)
#visual.ShowColorACloud(color_cloud, b'cloud')
cloud = pcl.PointCloud(pointcloud[:, :3])
visual.ShowMonochromeCloud(cloud)
#b = bev_utils.removePoints(pointcloud, cnf.boundary)
#rgb_map = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
#RGB_Map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
#RGB_Map[:, :, 2] = rgb_map[0, :, :]  # r_map
#RGB_Map[:, :, 1] = rgb_map[1, :, :]  # g_map
#RGB_Map[:, :, 0] = rgb_map[2, :, :]  # b_map
#RGB_Map *= 255
#RGB_Map = RGB_Map.astype(np.uint8)
#cv2.imshow("RGB map", RGB_Map)
cv2.imshow("Raw image", Raw_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#'''

#'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# video or camera set up (width = 640 height = 360)
source_w = 640
source_h = 360

# for obstacle trajectory
hov = 82.1 # horizontal of view
degree_over_pixel = hov / source_w # 1 pixel = how much degree
maximum_depth = 10
pts_old = []

# label
back_label = 0
road_label = 1
sidewalk_label = 2
person_label = 3
hole_label = 6
car_label = 5

fast_scnn = FastSCNN(device)

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
input_type = sl.InputType()
#if len(sys.argv) >= 2 :
#    input_type.set_from_svo_file(sys.argv[1])
input_type.set_from_svo_file("./ZED_raw_video_2020_05_27_12_20_13.svo")
init = sl.InitParameters(input_t=input_type)
init.camera_resolution = sl.RESOLUTION.HD720
init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init.coordinate_units = sl.UNIT.METER
init.depth_minimum_distance = 0.15 # Set the minimum depth perception distance to 15cm
init.depth_maximum_distance = 100

# Open the camera
err = zed.open(init)
if err != sl.ERROR_CODE.SUCCESS :
    print(repr(err))
    zed.close()
    exit(1)

# Set runtime parameters after opening the camera
runtime = sl.RuntimeParameters()
runtime.sensing_mode = sl.SENSING_MODE.FILL

# Get zed calibration parameters
calib = zed.get_camera_information().calibration_parameters.left_cam
zed_fx = round(calib.fx, 2)
zed_fy = round(calib.fy, 2)
zed_v_fov = round(calib.v_fov, 2)
zed_h_fov = round(calib.h_fov, 2)
zed_cx = int(calib.cx)
zed_cy = int(calib.cy)
print(f"CalibrationParameters(fx, fy, v_fov, v_fov, cx, cy) = {zed_fx}, {zed_fy}, {zed_v_fov}, {zed_h_fov}, {zed_cx}, {zed_cy}")

# Prepare new image size to retrieve half-resolution images
image_size = zed.get_camera_information().camera_resolution
image_size.width = image_size.width /2
image_size.height = image_size.height /2

# Declare your sl.Mat matrices
image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
image_zed_r = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
point_cloud = sl.Mat()

key = ' '
cnt = 0
v = True
visual = pcl.pcl_visualization.CloudViewing()
while key != 113 and v:
    err = zed.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS :
        # Retrieve the left image, depth image in the half-resolution
        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        zed.retrieve_image(image_zed_r, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)
        zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
        # Retrieve the RGBA point cloud in half resolution
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

        # To recover data from sl.Mat to use it with opencv, use the get_data() method
        # It returns a numpy array that can be used as a matrix with opencv
        image_ocv = image_zed.get_data()
        image_ocv_r = image_zed_r.get_data()
        depth_image_ocv = depth_image_zed.get_data()
        depth_image_ocv_ = (zed_fx)/depth_image_ocv
#        depth_image_ocv_ = (255 * (depth_image_ocv_ - np.min(depth_image_ocv_))) / (np.max(depth_image_ocv_) - np.min(depth_image_ocv_))
#        depth_image_ocv_ = depth_image_ocv_.astype(np.uint8)
        depth_image_ocv = cv2.applyColorMap(depth_image_ocv[:, :, :3].astype(np.uint8), cv2.COLORMAP_RAINBOW)
        depth_image_ocv_ = cv2.applyColorMap(depth_image_ocv_[:, :, :3].astype(np.uint8), cv2.COLORMAP_JET)
        
        point_cloud_np = point_cloud.get_data()
        
        Image_seg, result = fast_scnn.demo(image_ocv[:, :, :3])
        
        # rgb
        blue = image_ocv[:, :, 0].astype(np.uint32)
        green = image_ocv[:, :, 1].astype(np.uint32)
        red = image_ocv[:, :, 2].astype(np.uint32)
        
        # Segmentation
#        blue = Image_seg[:, :, 0].astype(np.uint32)
#        green = Image_seg[:, :, 1].astype(np.uint32)
#        red = Image_seg[:, :, 2].astype(np.uint32)
        
        rgb = np.left_shift(red, 16) + np.left_shift(green, 8) + np.left_shift(blue, 0)
        point_cloud_np[:, :, 3] = rgb.astype(np.float32)
        point_cloud_np = point_cloud_np.reshape(-1, 4)
        # change axis:y, z
        point_cloud_np[:, 1] = point_cloud_np[:, 1] * (-1)
        point_cloud_np[:, 2] = point_cloud_np[:, 2] * (-1)
        
#        # process the segmentation of road and sidewalk
#        binary_road = (cv2.inRange(result, (road_label, road_label, road_label), (road_label, road_label, road_label))) / 255
#        binary_sidewalk = (cv2.inRange(result, (sidewalk_label, sidewalk_label, sidewalk_label), (sidewalk_label, sidewalk_label, sidewalk_label))) / 255
#        binary_road = cv2.bitwise_or(binary_road, binary_sidewalk)
#        for i in range(len(binary_road)):
#            if i % 15 == 0:binary_road[i] = 0
#        binary_road = binary_road.astype(np.int).reshape(-1)
#        idx_road_label = np.where(binary_road == 1)
#        idx_road_label = idx_road_label[0]
#        point_cloud_np = np.delete(point_cloud_np, idx_road_label, axis=0)
        
        idx_nan_0 = np.where(np.isnan(point_cloud_np[:, 0]))
        point_cloud_np = np.delete(point_cloud_np, idx_nan_0, axis=0)
        idx_nan_1 = np.where(np.isnan(point_cloud_np[:, 1]))
        point_cloud_np = np.delete(point_cloud_np, idx_nan_1, axis=0)
        idx_nan_2 = np.where(np.isnan(point_cloud_np[:, 2]))
        point_cloud_np = np.delete(point_cloud_np, idx_nan_2, axis=0)
        
#        # x:0~80m
#        idx_0 = np.where(point_cloud_np[:, 0].reshape(-1) >= 80)
#        point_cloud_np = np.delete(point_cloud_np, idx_0, axis=0)
#        # y:0~1.25m
#        idx_1 = np.where(point_cloud_np[:, 1] >= 1.25)
#        point_cloud_np = np.delete(point_cloud_np, idx_1, axis=0)
#        # z:0~40m
#        idx_2 = np.where(point_cloud_np[:, 2] >= 40)
#        point_cloud_np = np.delete(point_cloud_np, idx_2, axis=0)
        
        x, y, z, col = point_cloud_np[:, 0], point_cloud_np[:, 1], point_cloud_np[:, 2], point_cloud_np[:, 3]
        point_cloud_np_ = np.zeros_like(point_cloud_np)
#        
        point_cloud_np_[:, 0] = -z
        point_cloud_np_[:, 1] = -x
        point_cloud_np_[:, 2] = -y
        
        print(point_cloud_np_.shape)
        
        color_cloud = pcl.PointCloud_PointXYZRGBA(point_cloud_np)
        visual.ShowColorACloud(color_cloud, b'ZED')
#        cloud = pcl.PointCloud(point_cloud_np[:, :3])
#        visual.ShowMonochromeCloud(cloud)
        v=not(visual.WasStopped())
#        b = bev_utils.removePoints(point_cloud_np_, cnf.boundary)
#        
#        b = point_cloud_np
#        rgb_map = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
#        RGB_Map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
#        RGB_Map[:, :, 2] = rgb_map[0, :, :]  # r_map
#        RGB_Map[:, :, 1] = rgb_map[1, :, :]  # g_map
#        RGB_Map[:, :, 0] = rgb_map[2, :, :]  # b_map
#        RGB_Map *= 255
#        RGB_Map = RGB_Map.astype(np.uint8)
        
        cv2.imshow("Image", image_ocv[:, :, :3])
        cv2.imshow("image_ocv_r", image_ocv_r[:, :, :3])
        cv2.imshow("Depth", depth_image_ocv)
        cv2.imshow("Depth_", depth_image_ocv_)
#        cv2.imshow("rgb_map", RGB_Map)
        
        cv2.waitKey(0)
#        if cnt > 10:break
        cnt += 1
        key = cv2.waitKey(1)

cv2.destroyAllWindows()
zed.close()

print("\nFINISH")

#'''