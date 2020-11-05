import numpy as np

class Plane:
    """docstring for Plane."""

    def __init__(self, pcd, row, col, no_points_per_cell):
        self.pcd = pcd
        self.row = row
        self.col = col

        self.bin = -1

        self.polar = 0
        self.azimuth = 0

        self.min_nonzero_points = no_points_per_cell * 0.5
        self.no_nonzero_points = (pcd[:,:,2] > 0).sum()
        self.planar = True
        self.max_depth_discontinuity = 100 # cm
        self.cell_diameter_trunctuated = 0

        self.DEPTH_SIGMA_COEFF =  0.000001425;
        self.DEPTH_SIGMA_MARGIN =  10;

        self.x_mean = np.mean(self.pcd[:,:,0])
        self.y_mean = np.mean(self.pcd[:,:,1])
        self.z_mean = np.mean(self.pcd[:,:,2])

        self.x_acc = self.pcd[:,:,0].sum()
        self.y_acc = self.pcd[:,:,1].sum()
        self.z_acc = self.pcd[:,:,2].sum()
        self.xx_acc = np.multiply(self.pcd[:,:,0],self.pcd[:,:,0]).sum()
        self.yy_acc = np.multiply(self.pcd[:,:,1],self.pcd[:,:,1]).sum()
        self.zz_acc = np.multiply(self.pcd[:,:,2],self.pcd[:,:,2]).sum()
        self.xy_acc = np.multiply(self.pcd[:,:,0],self.pcd[:,:,1]).sum()
        self.xz_acc = np.multiply(self.pcd[:,:,0],self.pcd[:,:,2]).sum()
        self.yz_acc = np.multiply(self.pcd[:,:,1],self.pcd[:,:,2]).sum()

        # check nonzero number of points threshold value
        if self.no_nonzero_points < self.min_nonzero_points:
            #print("Failed number of points")
            self.planar = False
            return

        # Check for depth discontinuities

        ## horizontal scan
        no_hori_discontinuities = 0
        for j in range(pcd.shape[1]):
            if j > 0 and j < pcd.shape[1]-1:
                z = pcd[int(pcd.shape[0]/2)][j][2]
                z_previous = pcd[int(pcd.shape[0]/2)][j-1][2]

                if z > 0 and abs(z-z_previous) > self.max_depth_discontinuity:
                    no_hori_discontinuities += 1

        if no_hori_discontinuities > 1:
            #print("Failed horizontal scan")
            self.planar = False
            return

        ## vertical scan
        no_vert_discontinuities = 0
        for j in range(pcd.shape[0]):
            if j > 0 and j < pcd.shape[0]-1:
                z = pcd[j][int(pcd.shape[1]/2)][2]
                z_previous = pcd[j-1][int(pcd.shape[1]/2)][2]

                if z > 0 and abs(z-z_previous) > self.max_depth_discontinuity:
                    no_vert_discontinuities += 1

        if no_vert_discontinuities > 1:
            #print("Failed vertical scan")
            self.planar = False
            return

        if self.planar == True:
            self.fitPlane()
            if self.MSE > pow(self.DEPTH_SIGMA_COEFF*self.z_mean*self.z_mean+self.DEPTH_SIGMA_MARGIN,2):
                #print("Failed MSE")
                self.planar = False



    def fitPlane(self):

        self.cov = np.array([[self.xx_acc - self.x_acc*self.x_acc/self.no_nonzero_points, self.xy_acc - self.x_acc*self.y_acc/self.no_nonzero_points,  self.xz_acc - self.x_acc*self.z_acc/self.no_nonzero_points],
                            [self.xy_acc - self.x_acc*self.y_acc/self.no_nonzero_points, self.yy_acc - self.y_acc*self.y_acc/self.no_nonzero_points,  self.yz_acc - self.y_acc*self.z_acc/self.no_nonzero_points],
                            [self.xz_acc - self.x_acc*self.z_acc/self.no_nonzero_points,self.yz_acc - self.y_acc*self.z_acc/self.no_nonzero_points,  self.zz_acc - self.z_acc*self.z_acc/self.no_nonzero_points]])

        eigen_values, eigen_vectors = np.linalg.eigh(self.cov)
        e = eigen_values[0]
        v = eigen_vectors[:,0]
        self.MSE = e / self.no_nonzero_points

        self.direction = - (v[0]*self.x_mean+v[1]*self.y_mean+v[2]*self.z_mean)
        self.score = eigen_values[1] / eigen_values[0]

        if self.direction > 0:
            self.normal_x= v[0]
            self.normal_y= v[1]
            self.normal_z= v[2]
        else:
            self.normal_x= -v[0]
            self.normal_y= -v[1]
            self.normal_z= -v[2]
            self.direction = -self.direction

    def expand(self, plane):
        self.x_acc += plane.x_acc
        self.y_acc += plane.y_acc
        self.z_acc += plane.z_acc
        self.xx_acc += plane.xx_acc
        self.yy_acc += plane.yy_acc
        self.zz_acc += plane.zz_acc
        self.xy_acc += plane.xy_acc
        self.xz_acc += plane.xz_acc
        self.yz_acc += plane.yz_acc
        self.no_nonzero_points += plane.no_nonzero_points
