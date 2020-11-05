import numpy as np
import open3d as o3d
import cv2
import yaml
import os
from plane import Plane
import math

def depthToPCDMap(w, h, fx, fy, cx, cy):

    map = np.zeros((h,w,2))
    map.fill(1.0)
    for i in range(w):
        for j in range(h):
            map[j][i][0] = 1 * (i - cx) / fx
            map[j][i][1] = 1 * (j - cy) / fy

    return(map)

def depthToPCD(npArr, pcdMap):

    pcd = np.zeros((npArr.shape[0],npArr.shape[1],3))
    z = npArr
    x = np.multiply(pcdMap[:,:,0], npArr)
    y = np.multiply(pcdMap[:,:,1], npArr)
    pcd[:,:,0] = x
    pcd[:,:,1] = y
    pcd[:,:,2] = z

    return pcd

def visualizePCD(npArr):
    npArr = np.reshape(npArr, (-1,3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npArr)
    o3d.visualization.draw_geometries([pcd])

def organizePointCloudByCell(pcd, cell_size):

    no_hori_cell = int(pcd.shape[1] / cell_size)
    no_vert_cell = int(pcd.shape[0] / cell_size)
    no_cell = no_hori_cell * no_vert_cell

    out = np.zeros((no_cell, cell_size, cell_size, 3))
    k = 0    #print("Seed: (",x1, y1,")", " \t x2,y2: (",x2,y2,") \t", regions[y2][x2])

    for i in range(no_vert_cell):
        for j in range(no_hori_cell):
            cell = pcd[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]
            out[k] = cell
            k +=1

    return out

def regionGrowing(planes, seed, regions, current_region, x2, y2):

    Tn = math.pi / 12.0
    Td = 50.0

    x1 = seed.col
    y1 = seed.row

    # check if plane is already part of region
    if regions[y2][x2] != 0:
        return regions

    plane1 = planes[y1][x1]
    plane2 = planes[y2][x2]

    # check if plane is in planar and therefore in remaining planes for regions
    if plane2.planar == False:
        return regions

    # check if dot product of planes normals is more than Tn
    if np.dot(np.array([plane1.normal_x, plane1.normal_y, plane1.normal_z]),
                np.array([plane2.normal_x, plane2.normal_y, plane2.normal_z])) <= Tn:
        return regions

    # check if point-to-plane distance of plane2's centroid to plane1 is less than
    # Td
    if pow(plane1.normal_x*plane2.x_mean + plane1.normal_y*plane2.y_mean + plane1.normal_z*plane2.z_mean + plane1.direction,2)>plane2.cell_distance_trunctuated:
        return regions

    #print("Before: ", regions[y2][x2])
    regions[y2][x2] = current_region
    #print("After: ", regions[y2][x2])

    # Check 4 connectivity
    if x2 > 0:
        regions = regionGrowing(planes, seed, regions, current_region, x2-1, y2)
    if x2 < planes.shape[1]-1:
        regions = regionGrowing(planes, seed, regions, current_region, x2+1, y2)
    if y2 > 0:
        regions = regionGrowing(planes, seed, regions, current_region, x2, y2-1)
    if y2 < planes.shape[0]-1:
        regions = regionGrowing(planes, seed, regions, current_region, x2, y2+1)
    return regions

# setup phase

PATCH_SIZE = 20

## get intrinsics

rgb_intr_params = np.array([[5.4886723733696215e+02, 0., 3.1649655835885483e+02],
                            [0, 5.4958402532237187e+02, 2.2923873484682150e+02],
                            [0, 0, 1]])

ir_intr_params = np.array([[5.7592685448804468e+02, 0., 3.1515026356388171e+02],
                            [0., 5.7640791601093247e+02, 2.3058580662101753e+02],
                            [0, 0, 1]])


fx_ir = ir_intr_params[0][0]
fy_ir = ir_intr_params[1][1]
cx_ir = ir_intr_params[0][2]
cy_ir = ir_intr_params[1][2]

fx_rgb = rgb_intr_params[0][0]
fy_rgb = rgb_intr_params[1][1]
cx_rgb = rgb_intr_params[0][2]
cy_rgb = rgb_intr_params[1][2]

## get dimensions from of rgb img

img = cv2.imread("Data/tunnel/rgb_0.png")

h = img.shape[0]
w = img.shape[1]

### vertical and horizontal cell

no_hori_cell = int(w / PATCH_SIZE)
no_vert_cell = int(h / PATCH_SIZE)

print("H:", no_hori_cell, "    V: ", no_vert_cell)

depth = cv2.imread("Data/tunnel/depth_171.png", cv2.IMREAD_ANYDEPTH)
depth_w = depth.shape[1]
depth_h = depth.shape[0]
pcdMap = depthToPCDMap(depth_w, depth_h, fx_ir, fy_ir, cx_ir, cy_ir)

### cell map
cellmap = np.zeros((h, w))

for row in range(h):
    cell_row = row / PATCH_SIZE
    local_row = row % PATCH_SIZE

    for col in range(w):
        cell_col = col / PATCH_SIZE
        local_col = col / PATCH_SIZE
        cellmap[row][col] = (cell_row*no_hori_cell+cell_col)*PATCH_SIZE*PATCH_SIZE + local_row*PATCH_SIZE + local_col;

### eigen matrices for storing w,h,channels, one called cloudarray and one called cloudarrayorganized
### initialize CAPE class

# for all images do:

rootDir = "./Data/tunnel/"
p = 0
for dirName, subdirList, fileList in os.walk(rootDir):
    no_frames = int((len(fileList) - 4) / 2)

    for i in range(no_frames):
        if p > 10:
            break
        p += 1
        rgb_file = dirName + "rgb_" + str(i) + ".png"
        depth_file = dirName + "depth_" + str(i) + ".png"

        ## read rgb image
        rgb = cv2.imread(rgb_file)
        #rgb  = cv2.imread("./Data/tunnel/rgb_422.png")

        ## read depth image
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        #depth  = cv2.imread("./Data/tunnel/depth_422.png", cv2.IMREAD_ANYDEPTH)

        ## Compute pointcloud from depth image (ux, uv)
        pcd = depthToPCD(depth, pcdMap)
        visualizePCD(pcd)

        # Organize pointcloud according to cells
        pcd_organized = organizePointCloudByCell(pcd, PATCH_SIZE)

        # Step A: Plane fitting

        planes = []

        sin_cos_angle_4_merge = math.sqrt(1-pow(math.pi/12,2))
        MAX_MERGE_DIST = 50.0
        k = 0
        for i in range(no_vert_cell):
            for j in range(no_hori_cell):
                plane = Plane(pcd_organized[k], row=i, col=j, no_points_per_cell=PATCH_SIZE*PATCH_SIZE)
                if plane.planar == True:
                    cell_diameter = np.linalg.norm(plane.pcd[0][0]-plane.pcd[-1][0])
                    plane.cell_distance_trunctuated=pow(min(max(cell_diameter*sin_cos_angle_4_merge,20.0),MAX_MERGE_DIST),2)
                planes.append(plane)
                k += 1

        # Step B: Histogram
        spherical_coordinates = np.zeros((no_hori_cell*no_vert_cell,2))
        planar_arr = np.zeros((no_hori_cell*no_vert_cell))

        ## Represent normals as spherical coordinates
        i = 0
        for plane in planes:
            if plane.planar == True:
                n_proj_norm =  math.sqrt(plane.normal_x*plane.normal_x+plane.normal_y*plane.normal_y)
                plane.polar = math.acos(-plane.normal_z);
                plane.azimuth = math.atan2(plane.normal_x/n_proj_norm,plane.normal_y/n_proj_norm);
                #print("Pol: \t", plane.polar, "\t Azh: \t", plane.azimuth)
                spherical_coordinates[i] = np.array([plane.polar, plane.azimuth])
                planar_arr[i] = 1 # remember that this plane is planar

            i += 1


        ## Initialize Histogram
        no_of_bins_pr_coordinate = 20 # why?
        no_of_bins = no_of_bins_pr_coordinate * no_of_bins_pr_coordinate # why?
        no_of_planar_planes = 0
        histogram = np.zeros((no_of_bins))
        bins = np.zeros((planar_arr.shape[0]))

        for i in range(len(planes)):
            if planes[i].planar == True:
                X_q = (no_of_bins_pr_coordinate-1)*(planes[i].polar)/(math.pi)

                if X_q>0:
                    Y_q = ((no_of_bins_pr_coordinate-1)*(planes[i].azimuth) - math.pi) /(2* math.pi)
                else:
                    Y_q = 0

                bin  = int(Y_q*no_of_bins_pr_coordinate + X_q)
                bins[i] = bin
                histogram[bin] +=1
                planes[i].bin = bin
                no_of_planar_planes += 1


        # Cell-wise Region Growing

        planes_merged = []
        planes = np.array(planes)
        planes = np.reshape(planes, (no_vert_cell, no_hori_cell))
        regions = np.zeros_like(planes)
        current_region = 1

        cells_deleted = 0
        p = 0
        no_of_planar_planes_old = 0
        plane_labels = []
        while no_of_planar_planes > 0:
            seed_candidates = []

            max_bin_frequency = np.max(histogram)
            most_frequent_bin = np.argwhere(histogram == max_bin_frequency)[0][0]

            if max_bin_frequency > 0:
                for i in range(planes.shape[0]):
                    for j in range(planes.shape[1]):
                        if planes[i][j].bin == most_frequent_bin:
                            seed_candidates.append(planes[i][j])

            if len(seed_candidates) < 5:
                break

            # Select seed based on MSE

            MSE_min = 2000000000000
            seed_id = 0
            for i in range(len(seed_candidates)):
                if seed_candidates[i].MSE < MSE_min:
                    seed_id = i
                    MSE_min = seed_candidates[i].MSE

            regions = regionGrowing(planes, seed_candidates[seed_id], regions, current_region, seed_candidates[seed_id].col, seed_candidates[seed_id].row)

            # remove all seed_candidates that has already been labelled and merge
            # them into a new plane
            #print(regions)
            for i in range(regions.shape[0]):
                for j in range(regions.shape[1]):
                    if regions[i][j] == current_region:
                        seed_candidates[seed_id].expand(planes[i][j])

                        bin_index = (i*no_hori_cell)+j
                        bin = int(bins[bin_index])
                        bins[bin_index] = -1
                        histogram[bin] += -1
                        no_of_planar_planes -= 1


            seed_candidates[seed_id].fitPlane()

            if seed_candidates[seed_id].score > 100:
                planes_merged.append(seed_candidates[seed_id])
                plane_labels.append(current_region)
            current_region +=1
            if no_of_planar_planes_old == no_of_planar_planes:
                p += 1
            if p == 3:
                break
            no_of_planar_planes_old = no_of_planar_planes


        kernel_4_neighbour = np.array([[0,1,0],[1,1,1],[0,1,0]])
        kernel_4_neighbour = np.uint8(kernel_4_neighbour)
        kernel_8_neighbour = np.ones((3,3))
        kernel_8_neighbour = np.uint8(kernel_8_neighbour)

        k = 0
        plane_pcds = []
        for label in plane_labels:
            ones = np.zeros_like(planes)
            ones.fill(1)
            mask = np.zeros_like(planes)
            mask[regions == label] = ones[regions == label]
            mask = np.uint8(mask)

            mask_eroded = cv2.erode(mask,kernel_4_neighbour)
            if np.max(mask_eroded) == 0:
                continue # mask is completely eroded

            mask_dilated = cv2.dilate(mask, kernel_8_neighbour)
            mask_diff = mask_dilated-mask_eroded

            max_distance = 9 * planes_merged[k].MSE

            points = []
            for i in range(mask_diff.shape[0]):
                for j in range(mask_diff.shape[1]):
                    if mask_diff[i][j] == 1:

                        distances = planes[i][j].pcd[:,:,0].flatten()*planes_merged[k].normal_x + planes[i][j].pcd[:,:,1].flatten()*planes_merged[k].normal_y + planes[i][j].pcd[:,:,2].flatten()*planes_merged[k].normal_z + planes_merged[k].direction
                        pcd_mask = planes[i][j].pcd
                        pcd_mask = np.reshape(pcd_mask,(-1,3))

                        for l in range(len(distances)):
                            if pow(distances[l],2) < max_distance:
                                points.append(pcd_mask[l])

            points = np.array(points)

            for i in range(mask_eroded.shape[0]):
                for j in range(mask_eroded.shape[1]):
                    if mask_eroded[i][j] == 1:
                        pcd_mask = planes[i][j].pcd
                        pcd_mask = np.reshape(pcd_mask,(-1,3))
                        points = np.concatenate((points,pcd_mask), axis=0)

            plane_pcds.append(points)

        for pcd_arr in plane_pcds:
            #visualizePCD(pcd_arr)
            pass
