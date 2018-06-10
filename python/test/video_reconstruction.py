import argparse
import cv2
import numpy as np

import matcher
import solver
import utils
from rendering import render_pts_and_cams
from sfm import PinholeCamera, SFM


class VideoReconstruction(object):
    """
    Reconstruct 3D geometry from video
    """

    def __init__(self, input_video_path, output_video_path, frame_start, frame_end):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.sfm = None

    def _init_sfm(self, input_hw):
        cam = PinholeCamera(width=input_hw[1], height=input_hw[0], fx=input_hw[0],
                            fy=input_hw[0], cx=input_hw[1] / 2., cy=input_hw[0] / 2.)
        self.sfm = SFM(cam)

    def run(self):
        video_input = cv2.VideoCapture(self.input_video_path)
        fps = int(round(video_input.get(cv2.CAP_PROP_FPS)))
        cols = int(round(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)))
        rows = int(round(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_output = cv2.VideoWriter(self.output_video_path, fourcc, fps, (cols, rows))

        try:
            frame_idx = 0
            prev_frame = None
            while video_input.isOpened() and frame_idx < self.frame_end:

                _, frame = video_input.read()

                if frame_idx < self.frame_start:
                    frame_idx += 1
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                from PIL import Image
                #Image.fromarray(frame.astype('uint8')).show()
                import time
                #time.sleep(.1)
                if self.sfm is None:
                    self._init_sfm(frame.shape)

                self.sfm.update(frame)

                '''
                kp1, kp2, n_kp1, n_kp2 = matcher.find_matching_points(prev_frame, frame, show=True)
                # keep track of which correspondence maps to which color
                kp_to_color = {i: prev_frame[kp[1], kp[0]] for i, kp in enumerate(kp1)}
                # run the solver with the correspondences to generate a reconstruction
                camera_kps = np.stack([n_kp1, n_kp2], axis=0)
                camera_params, points_3d, camera_indices, point_indices, points_2d, focal_length = \
                    solver.get_solver_params(camera_kps)

                recon_camera_params, recon_3d_points, recon_focal_length, points_to_remove = \
                    solver.run_solver(camera_params, points_3d, camera_indices, point_indices,
                                      points_2d, focal_length)
                recon_colors = np.asarray([kp_to_color[i] for i in point_indices \
                    if point_indices[i] not in points_to_remove])
                recon_3d_points = np.asarray([point for (i, point) in enumerate(recon_3d_points)\
                    if i not in points_to_remove])

                if True:
                    render_pts_and_cams(recon_3d_points, recon_colors, recon_camera_params[:, 3:], 
                                        recon_camera_params[:, :3])
                '''

                result = None
                video_output.write(result)

                frame_idx += 1

                if (frame_idx % fps) == 0:
                    print('output is {} seconds'.format(frame_idx / fps))

        except KeyboardInterrupt:
            print('stopping')

        video_input.release()
        video_output.release()

        all_cam_pos = []
        all_cam_rvecs = []

        for cam_pos, cam_rot_mat in self.sfm.all_cam_params:
            all_cam_pos.append(np.squeeze(cam_pos))
            all_cam_rvecs.append(utils.mat2rvec(np.linalg.inv(cam_rot_mat)))
        all_cam_pos = np.asarray(all_cam_pos)
        all_cam_rvecs = np.asarray(all_cam_rvecs)

        all_pts_3d = np.concatenate(self.sfm.all_pts_3d, axis=0)
        for t, R in self.sfm.all_cam_params:
            render_pts_and_cams(np.ones_like(all_pts_3d), [None for _ in all_pts_3d], all_cam_pos, all_cam_rvecs)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input-video-path', default='nyc.mp4', help='where to find input video')
    parser.add_argument('--output-video-path', default=None, help='where to save output video')
    parser.add_argument('--frame-start', default=100, help='which frame to start on')
    parser.add_argument('--frame-end', default=120, help='which frame to end on')

    args = parser.parse_args()

    video_recon = VideoReconstruction(input_video_path=args.input_video_path,
                                      output_video_path=args.output_video_path,
                                      frame_start=args.frame_start,
                                      frame_end=args.frame_end)
    video_recon.run()

if __name__ == '__main__':
    main()
