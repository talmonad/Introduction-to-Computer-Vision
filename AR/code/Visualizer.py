import cv2
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, window_name="Warping Visualizer", show_window=False):
        self.window_name = window_name
        if show_window:
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

    def convert_to_cv2_keypoints(self, keypoints):
        """Convert a list of [x, y] coordinates to cv2.KeyPoint objects."""
        cv2_keypoints = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in keypoints]
        return cv2_keypoints

    def draw_keypoints(self, img, keypoints):
        try:
            keypoints_image = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        except:
            keypoints_image = cv2.drawKeypoints(img, self.convert_to_cv2_keypoints(keypoints), None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Display the image with keypoints
        plt.figure(figsize=(10, 10))
        plt.imshow(keypoints_image, cmap='gray')
        plt.title("Keypoints on Reference Image")
        plt.axis("off")
        plt.show()

    def draw_contours(self, img, imgpts, show=False):
        """Draw contours on the image."""
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
        if show:
            plt.imshow(img)
            plt.show(block=True)
        return img

    def draw_cube(self, template_processor, rvec, tvec, calibration_matrix, dist, frame_for_display):
        cube_3D = template_processor.get_cube()
        imgpts, _ = cv2.projectPoints(cube_3D, rvec, tvec, calibration_matrix, dist)
        distorted_imgpts = cv2.undistortPoints(imgpts, calibration_matrix, dist, None, calibration_matrix)
        return self.draw_contours(frame_for_display, distorted_imgpts)


    def verify_homography(self, template_processor, frame, H):
        # Define the template corners in its 2D coordinate system
        template_corners = np.array([
            [0, 0],
            [template_processor.template_img_rgb.shape[1] - 1, 0],
            [template_processor.template_img_rgb.shape[1] - 1,
             template_processor.template_img_rgb.shape[0] - 1],
            [0, template_processor.template_img_rgb.shape[0] - 1]
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Project the corners onto the frame using the homography
        projected_corners = cv2.perspectiveTransform(template_corners, H)

        # Visualize the projected corners on the frame
        for pt in projected_corners:
            cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 5, (0, 255, 0), -1)
        plt.imshow(frame)
        plt.show(block=True)

    def close(self):
        """Destroys the visualization window."""
        cv2.destroyWindow(self.window_name)