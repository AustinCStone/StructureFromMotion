from scipy.optimize import least_squares
import numpy as np
import time

from bundle_adjustment import bundle_adjustment_sparsity, fun, project


def get_solver_params(camera_kps, camera_params_initial_guess=None,
                      points_3d_initial_guess=None,
                      focal_length_initial_guess=None):
    """ Set up the initial solver params """

    # camera_params with shape (n_cameras, 6) contains initial estimates of parameters for all
    # cameras. First 3 components in each row form a rotation vector, next 3 components form a
    # translation vector
    n_cams = camera_kps.shape[0]
    camera_params = camera_params_initial_guess
    if camera_params is None:
        camera_params = np.zeros((n_cams, 6))
    # points_3d with shape (n_points, 3) contains initial estimates
    # of point coordinates in the world frame.
    n_points = camera_kps.shape[1]
    points_3d = points_3d_initial_guess
    if points_3d is None:
        points_3d = np.ones((n_points, 3))
    # camera_ind with shape (n_observations,) contains indices of
    # cameras (from 0 to n_cameras - 1) involved in each observation.
    camera_indices = []
    # point_ind with shape (n_observations,) contatins indices of
    # points (from 0 to n_points - 1) involved in each observation.
    point_indices = []
    for cam_index, cam_kp in enumerate(camera_kps):
        camera_indices.extend([cam_index for kp in cam_kp if kp is not None])
        point_indices.extend([i for i, kp in enumerate(cam_kp) if kp is not None])
    camera_indices = np.asarray(camera_indices)
    point_indices = np.asarray(point_indices)
    # points_2d with shape (n_observations, 2) contains
    # measured 2-D coordinates of points projected on images in each observations.
    points_2d = []
    for cam_kp in camera_kps:
        for keyp in cam_kp:
            if keyp is None:
                continue
            points_2d.append(keyp)
    points_2d = np.asarray(points_2d, np.float32)

    focal_length = focal_length_initial_guess
    if focal_length is None:
        focal_length = 1.

    return camera_params, points_3d, camera_indices, point_indices, points_2d, focal_length


def run_solver(camera_params, points_3d, camera_indices, point_indices, points_2d, focal_length,
               verbose=2):
    """ Run the optimization """

    n_cams = camera_params.shape[0]
    n_pts = points_3d.shape[0]
    if verbose:
        print("n_cameras: {}".format(n_cams))
        print("n_points: {}".format(n_pts))
        print("Total number of parameters: {}".format(6 * n_cams + 3 * n_pts + 1))
        print("Total number of residuals: {}".format(2 * n_pts))

    A = bundle_adjustment_sparsity(n_cams, n_pts, camera_indices, point_indices)
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel(), focal_length))
    f0 = fun(x0, n_cams, n_pts, camera_indices, point_indices, points_2d)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=verbose, x_scale='jac', ftol=1e-4,
                        method='trf', args=(n_cams, n_pts, camera_indices, point_indices,
                                            points_2d))
    t1 = time.time()
    if verbose:
        print("Optimization took {} seconds".format(t1 - t0))

    num_cam_params = camera_params.size
    recon_camera_params = np.reshape(res.x[:num_cam_params], camera_params.shape)
    recon_3d_points = np.reshape(res.x[num_cam_params:-1], points_3d.shape)
    recon_focal_length = res.x[-1]

    return recon_camera_params, recon_3d_points, recon_focal_length