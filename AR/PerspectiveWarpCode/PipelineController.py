import cv2
import os
import numpy as np
from collections import deque  # For efficient history management

GOOD_MATCHES_THRESHOLD = 5


class PipelineController:
    def __init__(self, template_processor, video_processor, feature_matcher, visualizer, image_warper, blending_alpha=0.7, history_size=100):
        self.template_processor = template_processor
        self.video_processor = video_processor
        self.feature_matcher = feature_matcher
        self.visualizer = visualizer
        self.image_warper = image_warper
        self.blending_alpha = blending_alpha
        self.history_size = history_size  # Maximum size of history
        self.h_coarse_history = deque(maxlen=history_size)  # History of H matrices
        self.h_fine_tuned_history = deque(maxlen=history_size)  # History of H matrices

    def calculate_median_h(self, history):
        """Calculate the median homography matrix from the history."""
        if not history:
            return None
        # Stack history and calculate the median along the 0th axis
        return np.median(np.array(history), axis=0)

    def smooth_h(self, history, alpha=0.8):
        if not history:
            return None

        median_H = self.calculate_median_h(history)
        smooth_H = alpha*median_H + (1-alpha)*history[-1]

        return smooth_H

    def filter_out_bad_matches(self, src_pts_transformed, dst_pts, good_matches):
        xmin, xmax, ymin, ymax = self.image_warper.get_min_max(src_pts_transformed)
        final_good_matches = []
        for i, m in enumerate(good_matches):
            x, y = dst_pts[i]
            if xmin <= x <= xmax and ymin <= y <= ymax:
                final_good_matches.append(m)
        return final_good_matches

    def process_video(self):
        self.video_processor.initialize()
        count = 0

        while self.video_processor.cap.isOpened():
            ret, frame = self.video_processor.cap.read()
            if not ret:
                break

            frame_kp, good_matches = self.feature_matcher.match_features(frame)
            if good_matches is None or len(good_matches) <= GOOD_MATCHES_THRESHOLD:
                print(f"Frame {count}: No good matches, skipping...")
                continue

            # Initial homography calculation
            H, good_kp_template, good_kp_frame = self.feature_matcher.calculate_homography(
                self.template_processor.keypoints, frame_kp, good_matches, ransac_threshold=5.0
            )
            if H is None:
                print(f"Frame {count}: Homography couldn't be found, skipping...")
                continue
            self.h_coarse_history.append(H)
            smooth_H = self.smooth_h(self.h_coarse_history)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi_scale = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
            pts = self.image_warper.getPerspectivePoints(self.template_processor.template_img_rgb, smooth_H, roi_scale)
            good_matches = self.filter_out_bad_matches(pts, good_kp_frame, good_matches)
            if good_matches is None or len(good_matches) <= GOOD_MATCHES_THRESHOLD:
                print(f"Frame {count}: No good matches after filtering, skipping...")
                continue

            # Fine-tuned homography
            H_fine_tuned, good_kp_template, good_kp_frame = self.feature_matcher.calculate_homography(
                self.template_processor.keypoints, frame_kp, good_matches, ransac_threshold=2.0
            )
            if H_fine_tuned is None:
                print(f"Frame {count}: Fine-tuned H couldn't be found, skipping...")
                continue

            # Update history with fine-tuned H
            self.h_fine_tuned_history.append(H_fine_tuned)

            # Recalculate median H after fine-tuning
            smooth_H = self.smooth_h(self.h_fine_tuned_history)
            warped_frame = self.image_warper.warpTwoImages(
                frame_rgb, self.template_processor.template_img_rgb, smooth_H, roi_scale
            )
            blended_frame = cv2.addWeighted(frame, 1 - self.blending_alpha, warped_frame, self.blending_alpha, 0)

            # Save images and write video frame
            output_dir = self.video_processor.output_image_dir
            #cv2.imwrite(os.path.join(output_dir, f"frame_{count}_blended.jpg"), blended_frame)
            self.video_processor.out.write(blended_frame)

            count += 1

        self.video_processor.release()
        print("Processing complete.")