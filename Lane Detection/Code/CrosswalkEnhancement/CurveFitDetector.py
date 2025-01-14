import numpy as np
import cv2


class CurveFit:
    def __init__(self, lane_length_meters=30.5, lane_width_meters=3.7):
        """
        Initialize the class with real-world lane dimensions.
        """
        self.lane_length_meters = lane_length_meters  # Lane length (meters) corresponding to the image height
        self.lane_width_meters = lane_width_meters  # Lane width (meters) corresponding to the image width

    def draw_lanes(self, img, left_fit, right_fit, perspective_warp):
        """
        Draw the detected lane area onto the image.

        Parameters:
        - img: Original image on which the lane area will be drawn.
        - left_fit: Fitted polynomial coefficients for the left lane line.
        - right_fit: Fitted polynomial coefficients for the right lane line.
        - perspective_warp: Perspective warp object (must include `inv_perspective_warp` method).

        Returns:
        - Image with the lane area drawn.
        """
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        lane_overlay = np.zeros_like(img)

        # Generate points for the left and right lane lines
        left_points = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right_points = np.array(
            [np.flipud(np.transpose(np.vstack([right_fit, ploty])))]
        )
        lane_points = np.hstack((left_points, right_points))
        # Fill the lane area
        cv2.fillPoly(lane_overlay, np.int32([lane_points]), (0, 0, 255))

        # Warp the lane area back to the original perspective
        inv_perspective = perspective_warp.inv_perspective_warp(lane_overlay)

        # Blend the lane overlay with the original image
        output_img = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)

        return output_img

    def draw_lanes_as_lines(self, img, left_fit, right_fit, perspective_warp):
        """
        Draw the detected lane lines onto the image.

        Parameters:
        - img: Original image on which the lane lines will be drawn.
        - left_fit: Fitted polynomial coefficients for the left lane line.
        - right_fit: Fitted polynomial coefficients for the right lane line.
        - perspective_warp: Perspective warp object (must include `inv_perspective_warp` method).

        Returns:
        - Image with the lane lines drawn.
        """
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        lane_overlay = np.zeros_like(img)

        # Generate points for the left and right lane lines
        left_x = left_fit
        right_x = right_fit

        left_line_points = np.array(
            [np.transpose(np.vstack([left_x, ploty]))], dtype=np.int32
        )
        right_line_points = np.array(
            [np.transpose(np.vstack([right_x, ploty]))], dtype=np.int32
        )

        # Draw the left and right lane lines
        cv2.polylines(
            lane_overlay,
            [left_line_points],
            isClosed=False,
            color=(0, 0, 255),
            thickness=100,
        )
        cv2.polylines(
            lane_overlay,
            [right_line_points],
            isClosed=False,
            color=(255, 0, 0),
            thickness=100,
        )

        # Warp the lane overlay back to the original perspective
        inv_perspective = perspective_warp.inv_perspective_warp(lane_overlay)

        # Blend the lane overlay with the original image
        output_img = cv2.addWeighted(img, 0.8, inv_perspective, 1, 0)

        return output_img
