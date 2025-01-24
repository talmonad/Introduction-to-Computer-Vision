import cv2
import numpy as np
import matplotlib.pyplot as plt
class ImageProcessor:
    def __init__(self, template_path, visualizer):
        self.template_path = template_path
        self.template_img = None
        self.template_img_gray = None
        self.template_img_rgb = None
        self.keypoints = None
        self.descriptors = None
        self.sift = cv2.SIFT_create()
        self.visualizer = visualizer

    def process_template(self):
        self.template_img = cv2.imread(self.template_path)
        self.template_img_gray = cv2.cvtColor(self.template_img, cv2.COLOR_BGR2GRAY)
        self.template_img_rgb = cv2.cvtColor(self.template_img, cv2.COLOR_BGR2RGB)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(self.template_img_gray, None)
        #self.visualizer.draw_keypoints(self.template_img, self.keypoints)
        # # Extract keypoint coordinates
        # keypoint_coords = np.array([kp.pt for kp in self.keypoints])
        #
        # # Get the bounding box of keypoints
        # x_min, y_min = np.min(keypoint_coords[3:, :], axis=0).astype(int)
        # x_max, y_max = np.max(keypoint_coords[3:, :], axis=0).astype(int)
        #
        # # Add some padding around the bounding box
        # padding = 0  # Adjust this as needed
        # x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        # x_max, y_max = min(self.template_img.shape[1], x_max + padding), min(self.template_img.shape[0], y_max + padding)
        #
        # # Crop the reference image
        # cropped_ref = self.template_img[y_min:y_max, x_min:x_max]
        #
        # # Display the cropped image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(cropped_ref, cmap='gray')
        # plt.title("Cropped Reference Image Based on Keypoints")
        # plt.axis("off")
        # plt.show()


