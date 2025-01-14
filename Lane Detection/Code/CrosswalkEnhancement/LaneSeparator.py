import numpy as np
import cv2


def get_histogram(img):
    """Calculate the histogram of the lower half of the binary image."""
    return np.sum(img[img.shape[0] // 2 :, :], axis=0)


class LaneSeparator:
    def __init__(self):
        """Initialize lane parameters for smoothing."""
        self.left_fit_history = []
        self.right_fit_history = []
        self.smoothing_window = 20

    def sliding_window(
        self, img, nwindows=9, margin=150, minpix=1, draw_windows=True
    ):
        """
        Perform the sliding window approach to detect lane lines.

        Args:
            img: Binary warped image.
            nwindows: Number of sliding windows.
            margin: Width of the windows ± margin.
            minpix: Minimum number of pixels required to recenter a window.
            draw_windows: Flag to draw the sliding windows.

        Returns:
            out_img: Visualization image with detected lanes.
            lane_fits: Smoothed polynomial fits for left and right lanes.
            lane_lines: Coordinates of detected lane lines (left, right).
            ploty: Y-coordinates used for plotting.
        """
        # Prepare visualization image
        out_img = np.dstack((img, img, img)) * 255

        # Get histogram and base points for left and right lanes
        histogram = get_histogram(img)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Sliding window setup
        window_height = img.shape[0] // nwindows
        nonzero = img.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        leftx_current, rightx_current = leftx_base, rightx_base

        left_lane_inds, right_lane_inds = [], []

        # Step through each window
        for window in range(nwindows):
            # Define window boundaries
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw windows for visualization
            if draw_windows:
                cv2.rectangle(
                    out_img,
                    (win_xleft_low, win_y_low),
                    (win_xleft_high, win_y_high),
                    (100, 255, 255),
                    3,
                )
                cv2.rectangle(
                    out_img,
                    (win_xright_low, win_y_low),
                    (win_xright_high, win_y_high),
                    (100, 255, 255),
                    3,
                )

            # Identify nonzero pixels within windows
            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            # Append indices to lane lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # Recenter windows if sufficient pixels are found
            if len(good_left_inds) > minpix:
                leftx_current = np.int_(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int_(np.mean(nonzerox[good_right_inds]))

        # Concatenate arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract lane pixel positions
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        # Fit second-order polynomials
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            left_fit = np.median(self.left_fit_history, axis=0)
            right_fit = np.median(self.right_fit_history, axis=0)

        # Smooth polynomial fits using history
        self.update_fit_history(left_fit, right_fit)
        smoothed_left_fit = np.median(self.left_fit_history, axis=0)
        smoothed_right_fit = np.median(self.right_fit_history, axis=0)

        # Generate lane line coordinates
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = (
            smoothed_left_fit[0] * ploty**2
            + smoothed_left_fit[1] * ploty
            + smoothed_left_fit[2]
        )
        right_fitx = (
            smoothed_right_fit[0] * ploty**2
            + smoothed_right_fit[1] * ploty
            + smoothed_right_fit[2]
        )

        # Color lane pixels in the output image
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [
            255,
            0,
            100,
        ]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
            0,
            100,
            255,
        ]

        return (
            out_img,
            (left_fitx, right_fitx),
            (smoothed_left_fit, smoothed_right_fit),
            ploty,
        )

    def update_fit_history(self, left_fit, right_fit):
        """Update the history of lane fits for smoothing."""
        if len(self.left_fit_history) >= self.smoothing_window:
            self.left_fit_history.pop(0)
        if len(self.right_fit_history) >= self.smoothing_window:
            self.right_fit_history.pop(0)

        self.left_fit_history.append(left_fit)
        self.right_fit_history.append(right_fit)
