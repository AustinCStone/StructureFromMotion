""" Test that we can reconstruct 3D geometry from just point correspondences
on a pair of images. """

from PIL import Image
from scipy.optimize import least_squares
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import time

from bundle_adjustment import bundle_adjustment_sparsity, fun, project
from rendering import render_pts_and_cams
import utils
import matcher


def get_solver_params(camera_kps, rows, cols, colors,
                      camera_params_initial_guess=None,
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
    points_2d = np.asarray(points2d, np.float32)

    focal_length = focal_length_initial_guess
    if focal_length is None:
        focal_length = 1.

    return camera_params, points_3d, camera_indices, point_indices, points_2d, focal_length


def _run_solver(camera_params, points_3d, camera_indices, point_indices, points_2d, focal_length):
    """ Run the optimization """

    n_cams = camera_params.shape[0]
    n_pts = points_3d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(6 * n_cams + 3 * n_points + 1))
    print("Total number of residuals: {}".format(2 * n_pts))

    A = bundle_adjustment_sparsity(n_cams, n_pts, camera_indices, point_indices)
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel(), focal_length))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    num_cam_params = camera_params.size
    found_camera_params = np.reshape(res.x[:num_cam_params], camera_params.shape)
    found_3d_points = np.reshape(res.x[num_cam_params:-1], points_3d.shape)
    found_focal_length = res.x[-1]

    return found_camera_params, found_3d_points, found_focal_length


def check_reconstruction(gt_camera_params, rec_camera_params, gt_3d_points,
                         rec_3d_points, gt_focal_length, rec_focal_length,
                         rows, cols):
    
    rec_cam_rvecs = rec_camera_params[:, :3]
    rec_cam_positions = rec_camera_params[:, 3:6]
    cam_1_points2d = project(rec_3d_points,
                             rec_camera_params[np.asarray([0 for _ in rec_3d_points])],
                             rec_focal_length, rec_focal_length)

    cam_2_points2d = project(rec_3d_points,
                             rec_camera_params[np.asarray([1 for _ in rec_3d_points])],
                             rec_focal_length, rec_focal_length)

    cam_1_img = draw_points2d(cam_1_points2d, colors, rows, cols, show=True)
    cam_2_img = draw_points2d(cam_2_points2d, colors, rows, cols, show=True)

    render_pts_and_cams(rec_3d_points, colors, rec_cam_positions, rec_cam_rvecs)


def draw_points2d(points_2d, colors, rows, cols, show=False):
    img = np.zeros((rows, cols, 3))
    img_space_points_2d = (points_2d + 1.) / 2. * np.asarray((cols, rows))
    img_space_points_2d = img_space_points_2d.astype(np.int32)
    for (point_2d, color) in zip(img_space_points_2d, colors):
        if (0 <= point_2d[1] < rows) and (0 <= point_2d[0] < cols):
            img[point_2d[1], point_2d[0]] = color
    if show:
        Image.fromarray((img * 255.).astype('uint8'), 'RGB').show()
        time.sleep(.25)
    return img


def get_points():
    """ Generate some sample points """

    point1 = np.asarray([1, 1, 10], np.float32)
    point2 = np.asarray([1, 3, 12], np.float32)
    point3 = np.asarray([-1, -1, 5], np.float32)
    point4 = np.asarray([-2, 2, 8], np.float32)
    point5 = np.asarray([-5, 5, 15], np.float32)
    point6 = np.asarray([5, -5, 12], np.float32)
    point7 = np.asarray([1, -3, 8], np.float32)
    point8 = np.asarray([0, 0, 8], np.float32)
    point9 = np.asarray([3, 3, 9], np.float32)
    point10 = np.asarray([2, 2, 9], np.float32)

    points = np.stack([point1, point2, point3, point4, point5, point6, point7, point8,
                       point9, point10])

    random_state = np.random.RandomState(seed=42)
    point_colors = [random_state.rand(3) for _ in points]

    return points, point_colors


def get_cameras():
    """ Generate two sample cameras params """

    # the image rows and columns in pixels
    rows = 100
    cols = 100

    # the only camera intrinsics we use are focal_x and focal_y.
    # assuming a linear camera with the same intrinsics for each image
    focal_x = 1.
    focal_y = 1.

    # the orientation and position in world coords of the first camera
    cam_1_pos = np.asarray([0, 0, 0], np.float32)
    cam_1_rvec = utils.euler2rvec(0, 0, 0)

    # the orientation and position in world coords of the second camera
    cam_2_pos = np.asarray([1, 0, 0], np.float32)
    cam_2_rvec = utils.euler2rvec(.25, 0, 0)

    # camera_params with shape (n_cameras, 6) contains extrinsics for all cameras
    # first 3 components in each row form a rotation vector, next 3 components
    # form the camera position in 3 space
    camera_params = np.empty((2, 6)) 
    camera_params[0, :3] = cam_1_rvec
    camera_params[0, 3:] = cam_1_pos
    camera_params[1, :3] = cam_2_rvec
    camera_params[1, 3:] = cam_2_pos

    return camera_params, focal_x, focal_y, rows, cols


def test_ten_point(render_ground_truth=False, render_reconstruction=False):
    """ A simple test to check if we can recover the 3D positions of 10 known 3D points
    and camera parameters given two images of the points where the correspondences
    are known to be correct. """

    points, colors = get_points()

    camera_params, focal_x, focal_y, rows, cols = get_cameras()

    cam_1_points2d = project(points, camera_params[np.asarray([0 for _ in points])],
                             focal_x, focal_y)

    cam_2_points2d = project(points, camera_params[np.asarray([1 for _ in points])],
                             focal_x, focal_y)

    cam_1_img = draw_points2d(cam_1_points2d, colors, rows, cols, show=True)
    cam_2_img = draw_points2d(cam_2_points2d, colors, rows, cols, show=True)

    locs1, locs2 = matcher.find_matching_points_mock(utils.preprocess_img(cam_1_img),
                                                     utils.preprocess_img(cam_2_img))

    assert len(locs1) == len(locs2) == len(points)

    if render_ground_truth:
        render_pts_and_cams(points, colors, camera_params[:, 3:], camera_params[:, :3])

    solve(locs1, locs2, rows, cols, colors)

if __name__ == '__main__':
    test_ten_point(render_ground_truth=False,
                   render_reconstruction=False)
