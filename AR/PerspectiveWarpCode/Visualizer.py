import cv2
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, window_name="Warping Visualizer"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL) # Enables window resizing
        self.scale_factor = 0.7 # Adjust for smaller display

    def show_results(self, template_img, frame, warped_frame):
        """Displays the original frame, the template, and the warped result."""

        # Resize images to fit on screen
        template_img = cv2.resize(template_img, None, fx=self.scale_factor, fy=self.scale_factor)
        frame = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor)
        warped_frame = cv2.resize(warped_frame, None, fx=self.scale_factor, fy=self.scale_factor)


        # Resize images to match heights for concatenation, before concatenation, use height of the template
        h_template = template_img.shape[0]
        frame = cv2.resize(frame, (int(frame.shape[1] * h_template/frame.shape[0]), h_template))
        warped_frame = cv2.resize(warped_frame, (int(warped_frame.shape[1] * h_template/warped_frame.shape[0]), h_template))

        # Create a single image by concatenating the images horizontally
        vis = np.concatenate((template_img, frame, warped_frame), axis=1)

        cv2.imshow(self.window_name, vis)

    def draw_matches(self, template_img, frame, template_kp, frame_kp, good_matches):
        matches_image = cv2.drawMatches(
        template_img, template_kp,
        frame, frame_kp,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        matches_image_resized = cv2.resize(matches_image, (2*frame.shape[0], frame.shape[1]))
        cv2.imshow('Matches After Homography', matches_image_resized)
        cv2.waitKey(0)  # Wait for a keypress to close the window


    def draw_keypoints(self, img, keypoints):
        keypoints_image = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Display the image with keypoints
        plt.figure(figsize=(10, 10))
        plt.imshow(keypoints_image, cmap='gray')
        plt.title("Keypoints on Reference Image")
        plt.axis("off")
        plt.show()


    def close(self):
        """Destroys the visualization window."""
        cv2.destroyWindow(self.window_name)