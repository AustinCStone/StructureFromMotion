""" Dummy test """
import numpy as np
import copy
import time
from PIL import Image

import transformations as trans
from rendering import SphereCloud, OrientedRectangles, Renderer


def form_intrinsic_matrix(focal_x, focal_y):
    """ Form intrinsic matrix assuming no skew and pp in the image center """

    mat = np.eye(4)
    mat[0, 0] = focal_x
    mat[1, 1] = focal_y
    return mat


def identify_corresp(img1, img2, max_pix_movement=50):
    """ Find corresponding points between two images """

    loc1 = np.where(img1 != 0)
    loc2 = np.where(img2 != 0)

    return loc1, loc2


def recover_mesh(imgs):
    """ Recover a 3D mesh from a sequence of video frames """

    assert len(imgs)
    prev_img = imgs[0]
    for img in imgs[1:]:
        loc_prev, loc_next = identify_corresp(prev_img, img)
        prev_img = img


def render(points, point_colors, camera_positions, camera_rot_mats):
    """ Render a set of points and camera positions and orientations """

    # ugh
    points = points.transpose((1, 0))[:, :-1]
    camera_rot_mats = [c[:-1, :-1] for c in camera_rot_mats]

    pc = SphereCloud()
    normalize_factor = max(abs(points).max(), abs(camera_positions).max())
    normed_points = copy.deepcopy(points) / normalize_factor
    normed_camera_positions = copy.deepcopy(camera_positions) / normalize_factor

    for point, color in zip(normed_points, point_colors):
        pc.add_object(point, color=color)

    orr = OrientedRectangles()
    for pos, mat in zip(normed_camera_positions, camera_rot_mats):
        orr.add_rect(pos, mat)

    renderer = Renderer([pc, orr])
    renderer.run()


def test_six_point():
    """ See if we can recover the positions of
    6 known 3D points. """

    point1 = np.asarray([1, 1, 10, 1], np.float32)
    point2 = np.asarray([1, 3, 12, 1], np.float32)
    point3 = np.asarray([-1, -1, 5, 1], np.float32)
    point4 = np.asarray([-2, 2, 8, 1], np.float32)
    point5 = np.asarray([-5, 5, 15, 1], np.float32)
    point6 = np.asarray([5, -5, 12, 1], np.float32)

    points = np.stack([point1, point2, point3, point4, point5, point6])
    points = points.transpose((1, 0))

    def project_points(points, cam_pos, cam_rot_mat, colors):
        """ Project the points into the camera image """

        cam_pos_mat = trans.translation_matrix(-cam_pos)
        cam_extrinsic = np.matmul(np.linalg.inv(cam_rot_mat), cam_pos_mat)

        focal_x = 1.
        focal_y = 1.

        perspective_mat = form_intrinsic_matrix(focal_x, focal_y)[:-1, :]
        perspective_mat[:, -1] = 0

        rows = 100
        cols = 100

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

    random_state = np.random.RandomState(seed=42)
    colors = [random_state.rand(3) for _ in range(6)]

    cam_1_pos = np.asarray([0, 0, 0], np.float32)
    cam_1_rot_mat = trans.euler_matrix(0, 0, 0)
    cam_1_img = project_points(points, cam_1_pos, cam_1_rot_mat, colors)

    cam_2_pos = np.asarray([0, 0, 0], np.float32)
    cam_2_rot_mat = trans.euler_matrix(.5, 0, 0)
    cam_2_img = project_points(points, cam_2_pos, cam_2_rot_mat, colors)

    Image.fromarray((cam_1_img * 255.).astype('uint8'), 'RGB').show('cam 1')
    time.sleep(1.)
    Image.fromarray((cam_2_img * 255.).astype('uint8'), 'RGB').show('cam 2')

    render(points, colors, np.stack([cam_1_pos, cam_2_pos]),
           np.stack([cam_1_rot_mat, cam_2_rot_mat]))

if __name__ == '__main__':
    test_six_point()
