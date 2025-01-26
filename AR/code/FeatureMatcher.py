import cv2
import numpy as np


class FeatureMatcher:
    def __init__(self, template_kp, template_des):
        self.template_kp = template_kp
        self.template_des = template_des
        self.matcher = cv2.BFMatcher()
        self.sift = cv2.SIFT_create()

    def match_features(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = self.sift.detectAndCompute(gray_frame, None)
        if frame_des is None:
            return None, None, None

        matches = self.matcher.knnMatch(self.template_des, frame_des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]
        return frame_kp, good_matches

    def calculate_homography(self, template_kp, frame_kp, good_matches, ransac_threshold):
        good_kp_template = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        good_kp_frame = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        H, mask = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, ransac_threshold)
        return H, mask, good_kp_template, good_kp_frame