import cv2
import numpy as np
import matplotlib.pyplot as plt
class ImageProcessor:
    def __init__(self, template_path, visualizer, cb):
        self.template_path = template_path
        self.template_img = None
        self.template_img_gray = None
        self.template_img_rgb = None
        self.template_img_undistorted = None
        self.keypoints = None
        self.descriptors = None
        self.sift = cv2.SIFT_create()
        self.visualizer = visualizer
        self.cb = cb
        self.template_width_cm = 9
        self.template_height_cm = 16

    def get_cube(self):

        # Scale factor to reduce the size
        scale_factor = 0.2

        # Scaled dimensions
        scaled_width = self.template_width_cm * scale_factor
        scaled_height = self.template_height_cm * scale_factor
        scaled_depth = self.template_width_cm * scale_factor  # Assuming the cube's depth matches the width

        # Centering offsets
        offset_x = (self.template_width_cm - scaled_width) / 2
        offset_y = (self.template_height_cm - scaled_height) / 2

        # Define the scaled and centered cube points
        objectPoints = np.array(
            [
                [offset_x, offset_y, 0],  # Bottom face
                [offset_x, offset_y + scaled_height, 0],
                [offset_x + scaled_width, offset_y + scaled_height, 0],
                [offset_x + scaled_width, offset_y, 0],
                [offset_x, offset_y, -scaled_depth],  # Top face
                [offset_x, offset_y + scaled_height, -scaled_depth],
                [offset_x + scaled_width, offset_y + scaled_height, -scaled_depth],
                [offset_x + scaled_width, offset_y, -scaled_depth],
            ],
            dtype=np.float32
        )
        return objectPoints

    def get_3d_points(self, keypoints):
        template_height_pixels, template_width_pixels = self.template_img_gray.shape
        scale_x = self.template_width_cm / template_width_pixels
        scale_y = self.template_height_cm / template_height_pixels

        # Convert 2D points to 3D by scaling
        points_3d = []
        for pt in keypoints:
            x, y = pt
            points_3d.append([x * scale_x, y * scale_y, 0])  # Assume z = 0 for template points
        return np.array(points_3d, dtype=np.float32)

    def process_template(self, calibrate=False):
        self.template_img = cv2.imread(self.template_path)
        self.template_img_undistorted = self.cb.undistort_image(self.template_img) if calibrate else self.template_img
        self.template_img_gray = cv2.cvtColor(self.template_img_undistorted, cv2.COLOR_BGR2GRAY)
        self.template_img_rgb = cv2.cvtColor(self.template_img_undistorted, cv2.COLOR_BGR2RGB)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(self.template_img_gray, None)
        #self.visualizer.draw_keypoints(self.template_img_undistorted, self.keypoints)


