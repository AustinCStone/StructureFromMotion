from scipy.sparse import lil_matrix
import numpy as np


def rotate(points, rot_vecs):
    """ Rotate points by given rotation vectors.
    
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


def project(points, camera_params, focal_x, focal_y):
    """ Convert 3-D points to 2-D by projecting onto images. """
    points_proj = rotate(points, camera_params[:, :3])
    points_proj -= camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    points_proj *= np.asarray([focal_x, focal_y])
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:-1].reshape((n_points, 3))
    focal_x = params[-1]
    focal_y = params[-1]
    points_proj = project(points_3d[point_indices], camera_params[camera_indices],
                          focal_x, focal_y)
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):

    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3 + 1
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    # only the residuals for a particular camera
    # only have non-zero derivative wrt the params for their cam
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    # all residuals are impacted by (shared) focal length
    A[:, -1] = 1

    return A
