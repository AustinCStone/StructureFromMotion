""" Dummy test """
import numpy as np
import copy
import time
import cv2
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from PIL import Image

import transformations as trans
from rendering import SphereCloud, OrientedRectangles, Renderer
from bundle_adjustment import bundle_adjustment_sparsity, fun, project


def form_intrinsic_matrix(focal_x, focal_y):
    """ Form intrinsic matrix assuming no skew and pp in the image center """

    mat = np.eye(4)
    mat[0, 0] = focal_x
    mat[1, 1] = focal_y
    return mat


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def preprocess_img(img):
    assert img.min() >= 0.
    assert img.max() <= 255.

    # convert to (0-255) range if needed
    if img.max() <= 1.:
        img *= 255.

    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = rgb2gray(img)

    return img.astype('uint8')


def identify_corresp(img1, img2, max_pix_movement=50, normalize=True, show=False):
    """ Find corresponding points between two images """

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    if show:
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None,flags=2)
        plt.imshow(img3),plt.show()
    # Get the matching keypoints for each of the images

    list_kp1 = []
    list_kp2 = []
    for mat in matches:
        img1_idx = matches.queryIdx
        img2_idx = matches.trainIdx

        # x - columns
        # y - rows
        list_kp1.append(kp1[img1_idx].pt)
        list_kp2.append(kp2[img2_idx].pt)

        fundamental_matrix, mask = cv2.findFundamentalMat(
            previous_pts,
            current_pts,
            cv2.FM_RANSAC
        )

    return np.int32(list_kp1), np.int32(list_kp2)


def identify_corresp_mock(img1, img2, max_pix_movement=50, normalize=True, show=False):

    locs1 = np.where(img1 > 0)
    locs2 = np.where(img2 > 0)

    list_kp1 = []
    list_kp2 = []

    for i, val1 in enumerate(img1[locs1]):
        matched = False
        for j, val2 in enumerate(img2[locs2]):
            if val1 == val2:
                assert not matched
                matched = True
                list_kp1.append((locs1[1][i], locs1[0][i]))
                list_kp2.append((locs2[1][j], locs2[0][j]))

    # Draw first 10 matches.
    '''
    if show:
        cv_kp1 = [cv2.KeyPoint(x=p[0], y=p[1]) for p in list_kp1]
        cv_kp2 = [cv2.KeyPoint(x=p[0], y=p[1]) for p in list_kp2]
        matches = [cv2.Dmatch(distance=1., imgIdx=0, queryIdx=0, trainIdx=1)]
        img3 = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches, None, flags=2)
        plt.imshow(img3)
        plt.show()
    '''
    if normalize:
        kp1, kp2 = np.float32(list_kp1), np.float32(list_kp2)
        kp1 /= np.asarray([img1.shape[1], img1.shape[0]], np.float32)
        kp2 /= np.asarray([img2.shape[1], img2.shape[0]], np.float32)
        kp1 = kp1 * 2. - 1.
        kp2 = kp2 * 2. - 1.
        return kp1, kp2

    return np.int32(list_kp1), np.int32(list_kp2)





def recover_mesh(imgs):
    """ Recover a 3D mesh from a sequence of video frames """

    assert len(imgs)
    prev_img = imgs[0]
    for img in imgs[1:]:
        loc_prev, loc_next = identify_corresp(prev_img, img)
        prev_img = img


def render(points, point_colors, camera_positions, camera_rvecs):
    """ Render a set of points and camera positions and orientations """

    pc = SphereCloud()
    normalize_factor = max(abs(points).max(), abs(camera_positions).max())
    normed_points = copy.deepcopy(points) / normalize_factor
    normed_camera_positions = copy.deepcopy(camera_positions) / normalize_factor

    for point, color in zip(normed_points, point_colors):
        pc.add_object(point, color=color)

    orr = OrientedRectangles()
    for pos, rvec in zip(normed_camera_positions, camera_rvecs):
        orr.add_rect(pos, rvec)

    renderer = Renderer([pc, orr])
    renderer.run()


def solve(kp1, kp2, rows, cols, true_points3d, true_camera_params):
    # camera_params with shape (n_cameras, 8) contains initial estimates of parameters for all
    # cameras. First 3 components in each row form a rotation vector 
    # (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), next 3 components form a
    # translation vector, then a focal_x and focal_y
    camera_params = np.zeros((2, 8))
    camera_params[:, :3] = np.zeros(3) # stupid initial guess
    camera_params[:, 3:6] = np.zeros(3) # stupid initial guess
    camera_params[:, 6:] = 1.
    # points_3d with shape (n_points, 3) contains initial estimates
    # of point coordinates in the world frame.
    points_3d = np.ones((10, 3))
    points_3d = true_points3d
    camera_params = true_camera_params
    
    # camera_ind with shape (n_observations,) contains indices of
    # cameras (from 0 to n_cameras - 1) involved in each observation.
    camera_indices = np.arange(len(kp1) * 2)
    camera_indices = np.asarray([0 for _ in kp1] + [1 for _ in kp2])

    # point_ind with shape (n_observations,) contatins indices of
    # points (from 0 to n_points - 1) involved in each observation.
    point_indices = np.concatenate([np.arange(len(kp1)), np.arange(len(kp2))], axis=0)

    # points_2d with shape (n_observations, 2) contains
    # measured 2-D coordinates of points projected on images in each observations.
    points_2d = np.concatenate([kp1, kp2], axis=0)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    num_cam_params = camera_params.shape[0] * camera_params.shape[1]

    found_camera_params = np.reshape(res.x[:num_cam_params],
                                     camera_params.shape)

    found_3d_points = np.reshape(res.x[num_cam_params:], points_3d.shape)

    # Computing rotation matrix
    found_cam_rvecs = found_camera_params[:, :3]
    found_cam_positions = found_camera_params[:, 3:6]

    found_focal_x = found_camera_params[:, -2]
    found_focal_y = found_camera_params[:, -1]

    random_state = np.random.RandomState(seed=42)
    colors = [random_state.rand(3) for _ in range(n_points)]

    cam_1_points2d = project(found_3d_points, found_camera_params[np.asarray([0 for _ in found_3d_points])])
    cam_2_points2d = project(found_3d_points, found_camera_params[np.asarray([1 for _ in found_3d_points])])

    cam_1_img = draw_points2d(cam_1_points2d, colors, rows, cols, show=True)
    cam_2_img = draw_points2d(cam_2_points2d, colors, rows, cols, show=True)


    #render(found_3d_points, colors, found_cam_positions, found_cam_rvecs)

    '''
    plt.plot(res.fun)
    import ipdb
    ipdb.set_trace()
    plt.show()
    '''


def euler2mat(ai, aj, ak):

    i = 0
    j = 1
    k = 2

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.eye(3)
    
    M[i, i] = cj*ck
    M[i, j] = sj*sc-cs
    M[i, k] = sj*cc+ss
    M[j, i] = cj*sk
    M[j, j] = sj*ss+cc
    M[j, k] = sj*cs-sc
    M[k, i] = -sj
    M[k, j] = cj*si
    M[k, k] = cj*ci

    return M


def mat2euler(matrix, axes='sxyz'):

    i = 0
    j = 1
    k = 2

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    
    cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
    if cy > 1e-9:
        ax = math.atan2( M[k, j],  M[k, k])
        ay = math.atan2(-M[k, i],  cy)
        az = math.atan2( M[j, i],  M[i, i])
    else:
        ax = math.atan2(-M[j, k],  M[j, j])
        ay = math.atan2(-M[k, i],  cy)
        az = 0.0

    return ax, ay, az


def euler2rvec(a0, a1, a2):
    return np.squeeze(cv2.Rodrigues(euler2mat(a0, a1, a2))[0])


def rvec2euler(rvec):
    return mat2euler(cv2.Rodrigues(rvec)[0], axes='rzxy') 


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


def test_ten_point():
    """ See if we can recover the positions of
    6 known 3D points. """

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

    rows = 100
    cols = 100

    focal_x = 1.
    focal_y = 1.

    random_state = np.random.RandomState(seed=42)

    colors = [random_state.rand(3) for _ in points]

    cam_1_pos = np.asarray([0, 0, 0], np.float32)
    cam_1_rvec = euler2rvec(0, 0, 0)

    cam_2_pos = np.asarray([1., 0, 0], np.float32)
    cam_2_rvec = euler2rvec(.25, 0, 0)

    # camera_params with shape (n_cameras, 8) contains initial estimates of parameters for all
    # cameras. First 3 components in each row form a rotation vector, next 3 components
    # form the camera position in 3 space, then a focal_x and focal_y
    camera_params = np.empty((2, 8)) 
    camera_params[0, :3] = cam_1_rvec
    camera_params[0, 3:6] = cam_1_pos
    camera_params[1, :3] = cam_2_rvec
    camera_params[1, 3:6] = cam_2_pos
    camera_params[:, 6:] = np.asarray([focal_x, focal_y])
    cam_1_points2d = project(points, camera_params[np.asarray([0 for _ in points])])
    cam_2_points2d = project(points, camera_params[np.asarray([1 for _ in points])])

    cam_1_img = draw_points2d(cam_1_points2d, colors, rows, cols, show=True)
    cam_2_img = draw_points2d(cam_2_points2d, colors, rows, cols, show=True)

    locs1, locs2 = identify_corresp_mock(preprocess_img(cam_1_img), preprocess_img(cam_2_img))

    assert len(locs1) == len(locs2) == len(points)

    # render(points, colors, np.stack([cam_1_pos, cam_2_pos]), np.stack([cam_1_rvec, cam_2_rvec]))

    solve(locs1, locs2, rows, cols, points, camera_params)

if __name__ == '__main__':
    test_ten_point()
