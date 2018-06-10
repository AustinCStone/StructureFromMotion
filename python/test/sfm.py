import numpy as np 
import cv2

import utils

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
                k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]
        self.mat = np.zeros((3, 3))
        self.mat[0, 0] = fx; self.mat[0, 1] = 0.; self.mat[0, 2] = cx
        self.mat[1, 0] = 0.; self.mat[1, 1] = fy; self.mat[1, 2] = cy
        self.mat[2, 2] = 1.


class SFM:
    def __init__(self, cam, annotations=None):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.prev_R = None
        self.prev_t = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.pt_3d_ref = None
        self.pt_3d_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.all_pts_3d = []
        self.all_cam_params = []
        self.annotations = None
        if annotations is not None:
            with open(annotations) as f:
                self.annotations = f.readlines()

    def getAbsoluteScale(self, frame_id=None):  #specialized for KITTI odometry dataset
        if self.annotations is None or frame_id is None:
            return 1.
        ss = self.annotations[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

    def get3dPoints(self):
        # pts_l_norm = cv2.undistortPoints(np.expand_dims(self.px_ref, axis=1), cameraMatrix=self.cam.mat, distCoeffs=None)
        # pts_r_norm = cv2.undistortPoints(np.expand_dims(self.px_cur, axis=1), cameraMatrix=self.cam.mat, distCoeffs=None)

        M_l = np.hstack((self.prev_R, self.prev_t))
        M_r = np.hstack((self.cur_R, self.cur_t))
        P_l = np.dot(self.cam.mat,  M_l)
        P_r = np.dot(self.cam.mat,  M_r)
        point_4d_hom = cv2.triangulatePoints(P_l, P_r, self.px_ref.T, self.px_cur.T)
        #point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d_hom[:3, :] / point_4d_hom[3]
        #point_3d = np.linalg.inv(self.prev_R).dot(point_3d) + self.prev_t
        return point_3d.T

    def processFirstFrame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME 
        self.px_ref = self.px_cur

    def updateHistory(self):
        self.all_cam_params.append((self.cur_t, self.cur_R))
        self.all_pts_3d.append(self.pt_3d_cur)

    def processFrame(self, frame_id=None):
        self.prev_R, self.prev_t = self.cur_R, self.cur_t
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
        absolute_scale = self.getAbsoluteScale(frame_id)
        if(absolute_scale > 0.1):
            self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
            self.cur_R = R.dot(self.cur_R)
        if(self.px_ref.shape[0] < kMinNumFeature):
            print("Resetting features!")
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        else:
            self.pt_3d_cur = self.get3dPoints()
        self.px_ref = self.px_cur

    def update(self, img, frame_id=None):
        assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
            self.updateHistory()
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame
