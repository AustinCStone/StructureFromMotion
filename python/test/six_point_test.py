""" Dummy test """
import numpy as np
import copy

import transformations as trans
from rendering import SphereCloud, Renderer


def form_intrinsic_matrix(focal_x, focal_y):
    """ Form intrinsic matrix assuming no skew and pp in the image center """

    mat = np.identity(3)
    mat[0, 0] = focal_x
    mat[1, 1] = focal_y
    mat[0, 2] = pp_x
    mat[1, 2] = pp_y
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


def render(points, camera_positions, camera_orientations):
    """ Render a set of points and camera positions and orientations """

    pc = SphereCloud()
    normalize_factor = max(points.max(), abs(points.min()))

    normed_points = copy.deepcopy(points) / normalize_factor

    for point in normed_points:
        pc.add_object(point)
    renderer = Renderer([pc])
    renderer.run()


def test_six_point():
    """ See if we can recover the positions of
    6 known 3D points. """

    point1 = np.asarray([1, 1, 10], np.float32)
    point2 = np.asarray([1, 3, 12], np.float32)
    point3 = np.asarray([-1, -1, 5], np.float32)
    point4 = np.asarray([-2, 2, 8], np.float32)
    point5 = np.asarray([-5, 5, 15], np.float32)
    point6 = np.asarray([5, -5, 12], np.float32)

    points = np.stack([point1, point2, point3, point4, point5, point6])
    render_points(points)
    cam_1_pos = np.asarray([0, 0, 0], np.float32)
    # looking directly down z axis
    cam_1_orientation = trans.euler_matrix(0, 0, 0)

    cam_2_pos = np.asarray([2, 0, 0], np.float32)
    cam_2_orientation = trans.euler_matrix(0, 0, 0)

    focal_x = .5
    focal_y = .5

    intrinsic_mat = form_intrinsic_matrix(focal_x, focal_y)
    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)

    rows = 100
    cols = 100

    cam_1_coords = np.matmul(cam_1_orientation, points - cam_1_pos)

    cam_1_pixel_coords = np.dot(intrinsic_mat_inv, 
                                np.asarray([cam_1_coords[0] / cam_1_coords[2],
                                            cam_1_coords[1] / cam_1_coords[2],
                                            1]))[:2]

    cam_2_coords = np.dot(cam_2_orientation, point - cam_2_pos)

    cam_2_pixel_coords = np.dot(intrinsic_mat_inv,
                                np.asarray([cam_2_coords[0] / cam_2_coords[2],
                                            cam_2_coords[1] / cam_2_coords[2],
                                            1]))[:2]


    print(cam_1_pixel_coords * np.asarray((rows, cols), np.float32))
    print(cam_2_pixel_coords * np.asarray((rows, cols), np.float32))


if __name__ == '__main__':
    test_six_point()
