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