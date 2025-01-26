import cv2
import numpy as np
from numpy import float32, int32
from numpy import concatenate, array


class ImageWarper:
    def __init__(self):
        pass

    def getPerspectivePoints(self, src_img, H, roi_scale):
        h, w = src_img.shape[:2]
        # top_left, bottom_left, bottom_right, top_right
        pts = float32([[roi_scale[0][0]*w, roi_scale[0][1]*h], [roi_scale[1][0]*w, roi_scale[1][1]*h], [roi_scale[2][0]*w, roi_scale[2][1]*h], [roi_scale[3][0]*w, roi_scale[3][1]*h]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H)

    def get_min_max(self, pts):
        [xmin, ymin] = int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = int32(pts.max(axis=0).ravel() + 0.5)
        return xmin, xmax, ymin, ymax

    def warpTwoImages(self, img1, img2, H, roi_scale):
        # warp img2 to img1 with homography H
        h, w = img1.shape[:2]
        # top_left, bottom_left, bottom_right, top_right
        pts1 = float32([[roi_scale[0][0]*w, roi_scale[0][1]*h], [roi_scale[1][0]*w, roi_scale[1][1]*h], [roi_scale[2][0]*w, roi_scale[2][1]*h], [roi_scale[3][0]*w, roi_scale[3][1]*h]]).reshape(-1, 1, 2)
        pts2_ = self.getPerspectivePoints(img2, H, roi_scale)
        pts = concatenate((pts1, pts2_), axis=0)
        xmin, xmax, ymin, ymax = self.get_min_max(pts)
        t = [-xmin, -ymin]
        Ht = array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]]) # translate

        result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
        result[t[1]:h+t[1], t[0]:w+t[0]] = img1
        if result.shape != img1.shape:
            result = cv2.resize(result, (w, h), cv2.INTER_CUBIC)
        return result