from matplotlib import pyplot as plt
import cv2
import numpy as np


def find_matching_points(img1, img2, max_pix_movement=50, normalize=True, show=False):
    """ Find corresponding points between two images using openCV ORB detector """

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


def find_matching_points_mock(img1, img2, max_pix_movement=50, show=False):

    if show:
        raise NotImplementedError
        
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
    
    n_kp1, n_kp2 = np.float32(list_kp1), np.float32(list_kp2)
    n_kp1 /= np.asarray([img1.shape[1], img1.shape[0]], np.float32)
    n_kp2 /= np.asarray([img2.shape[1], img2.shape[0]], np.float32)
    n_kp1 = n_kp1 * 2. - 1.
    n_kp2 = n_kp2 * 2. - 1.

    return np.int32(list_kp1), np.int32(list_kp2), n_kp1, n_kp2