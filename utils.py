import numpy as np

# https://github.com/colmap/colmap/blob/main/scripts/python/read_write_dense.py
def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def equation_plane(pts):
    x1,y1,z1 = pts[0]
    x2,y2,z2 = pts[1]
    x3,y3,z3 = pts[2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)

    return np.array([a,b,c,d]) / d

def calculate_distances(pts, plane): 
     
    d = abs(pts @ plane[:3] + plane[3]) 
    e = np.linalg.norm(plane[:3])
    
    return d/e

def find_ground_plane(pts, max_iters=1000, t=1e-1):
    best_num = -1
    best_plane = None
    best_inliers = None
    for i in range(max_iters):
        indices = np.random.choice(pts.shape[0], size=3, replace=True)
        chosen_pts = pts[indices]
        plane = equation_plane(chosen_pts)

        distances = calculate_distances(pts, plane)
        inliers = np.nonzero(distances < t)[0]
        num_inliers = len(inliers)

        if num_inliers > best_num:
            best_num = num_inliers
            best_plane = plane
            best_inliers = inliers

    return best_plane, best_inliers, best_num


# https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix







import os
from PIL import Image



# 3d functions
# getting matrices
def get_extrinsic_matrix(pose):
    R = quaternion_rotation_matrix(pose[:4])
    T = pose[-3:]

    extrinsic = np.zeros((4,4))
    extrinsic[:3,:3] = R
    extrinsic[:3, -1] = T
    extrinsic[3,3] = 1

    return extrinsic

def get_K_matrix(cam_params):
    K = np.zeros((3,4))
    K[(0,1),(0,1)] = cam_params[0]
    K[:2, 2] = cam_params[1:3]
    K[2,2] = 1

    return K

def get_K_inv(K):
    kinv = np.zeros((4,3))
    f = K[0,0]
    kinv[(0,1),(0,1)] = 1/f
    kinv[2,2] = 1
    kinv[:2, 2] = -K[:2,2] / f
    return kinv
    

# getting view
def get_camera_view(points3d, extrinsic):
    camera_points = np.pad(points3d, pad_width=((0,0), (0,1)), mode="constant", constant_values=1) @ extrinsic.T
    return camera_points

def get_image_view(camera_points, intrinsic):
    pixel_points = camera_points @ intrinsic.T
    pixel_points = pixel_points / pixel_points[:,2:3]
    return pixel_points

def get_in_view_points(pixel_points):
    i_under = pixel_points[:,0] < 720
    i_over = pixel_points[:,0] > 0
    j_under = pixel_points[:,1] < 1280
    j_over = pixel_points[:,1] > 0
    pixel_inliers = i_under*i_over*j_under*j_over
    return pixel_inliers

# get indices of 3d points
def select_apple_points(camera_points, pixel_points, inliers, mask):
    apple_points_dict = dict()
    for idx, pt in zip(np.where(inliers)[0], np.floor(pixel_points[inliers,:2])):
        new_val = camera_points[idx,2]
        r,c = np.int32(pt)
        key = f"{r}-{c}"
        if apple_points_dict.get(key, np.inf) > new_val:
            r,c = np.int32(pt)
            if mask[c,r] == 1:
                apple_points_dict[key] = idx
    apple_points_indices = list(apple_points_dict.values())

    return apple_points_indices


# get 3d points
def get_3d_apple_points(points3d, extrinsic, intrinsic, mask):
    cam_pts = get_camera_view(points3d, extrinsic)
    pixel_pts = get_image_view(cam_pts, intrinsic)
    inliers = get_in_view_points(pixel_pts)

    apple_points_indices = select_apple_points(cam_pts, pixel_pts, inliers, mask)

    return apple_points_indices, pixel_pts




# map
class Maps():
    def __init__(self, filenames, mask_root, image_root) -> None:
        self.filenames = filenames
        self.mask_root = mask_root
        self.image_root = image_root

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        mask = np.array(Image.open(os.path.join(self.mask_root, self.filenames[idx])))
        bool_mask = mask > 0
        mask = np.int8(bool_mask)

        image = np.array(Image.open(os.path.join(self.image_root, self.filenames[idx])))

        return mask, image