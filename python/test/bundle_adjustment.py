from scipy.sparse import lil_matrix
import numpy as np


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]

    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj -= camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    points_proj *= camera_params[:, 6:]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 8].reshape((n_cameras, 8))
    points_3d = params[n_cameras * 8:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    points_proj = (points_proj + 1.) / 2.
    import ipdb
    ipdb.set_trace()
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):

    m = camera_indices.size * 2
    n = n_cameras * 8 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(8):
        A[2 * i, camera_indices * 8 + s] = 1
        A[2 * i + 1, camera_indices * 8 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 8 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 8 + point_indices * 3 + s] = 1

    return A

'''
# camera_params with shape (n_cameras, 9) contains initial estimates of parameters for all
# cameras. First 3 components in each row form a rotation vector 
# (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), next 3 components form a
# translation vector, then a focal distance and two distortion parameters.
camera_params = 
# points_3d with shape (n_points, 3) contains initial estimates
# of point coordinates in the world frame.
points_3d = 
# camera_ind with shape (n_observations,) contains indices of
# cameras (from 0 to n_cameras - 1) involved in each observation.
camera_indices = range(2)
# point_ind with shape (n_observations,) contatins indices of
# points (from 0 to n_points - 1) involved in each observation.
point_indices = 
# points_2d with shape (n_observations, 2) contains
# measured 2-D coordinates of points projected on images in each observations.
points_2d = 

n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
'''

'''
def project_points(points, cam_pos, cam_rot_mat, colors, focal_x, focal_y):
    """ Project the points into the camera image """

    cam_pos_mat = trans.translation_matrix(-cam_pos)
    cam_extrinsic = np.matmul(np.linalg.inv(cam_rot_mat), cam_pos_mat)

    perspective_mat = form_intrinsic_matrix(focal_x, focal_y)[:-1, :]
    perspective_mat[:, -1] = 0

    # transform points from world coordinates into camera centric coords, then
    # transform points into pixel coords
    cam_pixel_coords = np.matmul(perspective_mat, np.matmul(cam_extrinsic, points))
    cam_pixel_coords /= cam_pixel_coords[-1, :]
    cam_pixel_coords = (cam_pixel_coords[:-1, :] + 1.) / 2.
    cam_pixel_coords *= np.asarray([rows, cols]).reshape((2, 1))

    cam_img = np.zeros((rows, cols, 3))
    cam_pixel_coords = np.transpose(cam_pixel_coords, (1, 0))
    for color, pixel_coord in zip(colors, cam_pixel_coords):
        if 0 < pixel_coord[0] < cols and 0 < pixel_coord[1] < rows:
            cam_img[int(pixel_coord[1]), int(pixel_coord[0])] = color

    return cam_img
'''