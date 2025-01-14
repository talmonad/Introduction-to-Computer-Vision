import cv2
from Debugger import Debugger
import numpy as np
import matplotlib.pyplot as plt
from ImageWarping import flexible_warp_points


class CrosswalkDetector:
    def __init__(
        self,
        debug=False,
        output_dir="debug_outputs",
        spacing_range=(35, 100),
        parallel_tolerance=0.15,
        intensity_stripe_tolerance=(20, 100),
        valid_crosswalk_shape_ratio=1.8,
    ):
        """
        Initialize the CrosswalkDetector.

        Parameters:
            debug (bool): Enable debug mode for intermediate visualizations.
            output_dir (str): Directory to save debug outputs.
            spacing_range (tuple): Min and max spacing for valid crosswalk detection.
            parallel_tolerance (float): Tolerance for line slope comparison.
            intensity_stripe_tolerance (tuple): Min and max for average stripe spacing in intesity analyzation.
            valid_crosswalk_shape_ratio (float): A limit on w/h ratio to maintain a valid crosswalk shape.
        Note: Only use debugger when inspecting single frames !!
        """
        self.debug = debug
        self.debugger = Debugger(debug, output_dir)
        self.spacing_range = spacing_range
        self.parallel_tolerance = parallel_tolerance
        self.intensity_stripe_tolerance = intensity_stripe_tolerance
        self.valid_crosswalk_shape_ratio = valid_crosswalk_shape_ratio

    @staticmethod
    def calculate_slope(x1, y1, x2, y2):
        if x2 - x1 == 0:
            return float("inf")
        return (y2 - y1) / (x2 - x1)

    @staticmethod
    def analyze_intensity_profile(
        rect_crop,
        orientation="horizontal",
        stripe_tolerance=(20, 100),
        debug=False,
    ):
        """
        Analyze intensity profile to detect periodic patterns like crosswalk stripes.

        Parameters:
            rect_crop (ndarray): Cropped rectangle region (grayscale).
            orientation (str): 'horizontal' to analyze vertical stripes, 'vertical' for horizontal stripes.
            stripe_tolerance (tuple): Expected stripe spacing range (min_spacing, max_spacing) in pixels.

        Returns:
            is_striped (bool): True if periodic stripe patterns are detected.
        """
        # Compute intensity profile based on orientation
        if orientation == "horizontal":
            # Sum pixel intensities row-wise for vertical stripes
            profile = np.sum(rect_crop, axis=1)
        else:  # 'vertical'
            # Sum pixel intensities column-wise for horizontal stripes
            profile = np.sum(rect_crop, axis=0)

        # Normalize the profile (zero mean)
        profile = profile - np.mean(profile)
        if debug:
            # Plot the profile
            plt.figure(figsize=(10, 5))
            plt.plot(profile, label="Intensity Profile", color="blue")
            plt.title(f"Intensity Profile ({orientation.capitalize()})")
            plt.xlabel("Row index")
            plt.ylabel("Normalized Intensity")
            plt.axhline(
                0, color="red", linestyle="--", linewidth=1, label="Zero Line"
            )
            plt.legend()
            plt.grid()
            plt.show()
        # Detect zero-crossings in the intensity profile
        zero_crossings = np.where(np.diff(np.sign(profile)))[0]

        # Compute distances between zero-crossings (periods)
        distances = np.diff(zero_crossings)

        # Check if distances are consistent and within expected range
        if len(zero_crossings) > 2:
            avg_spacing = np.mean(distances)

            if stripe_tolerance[0] <= avg_spacing <= stripe_tolerance[1]:
                return True

        return False

    def detect_horizontal_lines(self, lines, threshold=50):
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < threshold:
                    horizontal_lines.append((x1, y1, x2, y2))
        return sorted(horizontal_lines, key=lambda l: l[1])

    def detect_parallel_pairs(self, horizontal_lines):
        parallel_pairs = []
        for i in range(len(horizontal_lines)):
            for j in range(i + 1, len(horizontal_lines)):
                x1, y1, x2, y2 = horizontal_lines[i]
                x3, y3, x4, y4 = horizontal_lines[j]

                slope1 = self.calculate_slope(x1, y1, x2, y2)
                slope2 = self.calculate_slope(x3, y3, x4, y4)

                if abs(slope1 - slope2) < self.parallel_tolerance:
                    parallel_pairs.append(
                        (horizontal_lines[i], horizontal_lines[j])
                    )
        return parallel_pairs

    def detect_potential_crosswalk(self, parallel_pairs):
        potential_crosswalk = []
        for i in range(len(parallel_pairs) - 1):
            (x1, y1, x12, y12), (x2, y2, x22, y22) = parallel_pairs[i]
            (x3, y3, x32, y32), (x4, y4, x42, y42) = parallel_pairs[i + 1]

            spacing = (abs(y1 - y3) + abs(y12 - y32)) / 2

            if self.spacing_range[0] <= spacing <= self.spacing_range[1]:
                left = min(x1, x12, x2, x22, x3, x32, x4, x42)
                right = max(x1, x12, x2, x22, x3, x32, x4, x42)
                top = min(y1, y12, y2, y22, y3, y32, y4, y42)
                bottom = max(y1, y12, y2, y22, y3, y32, y4, y42)
                potential_crosswalk.append((left, right, top, bottom))
        return potential_crosswalk

    def detect_crosswalk(
        self,
        frame,
        pre_processed_image,
    ):

        height, width = frame.shape[:2]
        src, dest = flexible_warp_points(
            (width * 0.4, height * 0.74),
            (width * 0.6, height * 0.74),
            (width * 0.25, height - 1),
            (width * 0.85, height - 1),
        )
        src = src.reshape((-1, 1, 2))

        # Debug: Save ROI visualization
        frame_with_roi = frame.copy()
        cv2.polylines(frame_with_roi, [np.int32(src)], True, (0, 0, 255), 3)
        self.debugger.save_image(frame_with_roi, "frame_with_roi.png")

        H, _ = cv2.findHomography(src, dest)
        Minv = np.linalg.inv(H)
        out = cv2.warpPerspective(
            pre_processed_image,
            H,
            (int(dest[2][0]), int(dest[2][1])),
            flags=cv2.INTER_LINEAR,
        )
        self.debugger.save_image(out, "warped_image.png")

        # Canny edge detection
        edges = cv2.Canny(out, 50, 150)
        self.debugger.save_image(edges, "canny_edges.png")

        # Line and crosswalk detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=40,
            maxLineGap=10,
        )
        horizontal_lines = self.detect_horizontal_lines(lines)
        parallel_pairs = self.detect_parallel_pairs(horizontal_lines)
        potential_crosswalk = self.detect_potential_crosswalk(parallel_pairs)

        most_probable_crosswalk = None
        debug_img = out.copy()
        for pcw in potential_crosswalk:
            left, right, top, bottom = pcw
            cropped = out[top:bottom, left:right]
            is_stripe = self.analyze_intensity_profile(
                cropped,
                debug=self.debug,
                stripe_tolerance=self.intensity_stripe_tolerance,
            )
            h, w = cropped.shape
            aspect_ratio = w / h

            if (
                not is_stripe
                or aspect_ratio < self.valid_crosswalk_shape_ratio
            ):
                continue

            most_probable_crosswalk = pcw

            cv2.rectangle(
                debug_img,
                (left, top),  # top-left corner
                (
                    # int(output_image.shape[1] * 0.9),
                    right,
                    bottom,
                ),  # bottom-right corner
                (0, 0, 255),  # color = RED (B, G, R)
                2,  # thickness = 2
            )

            # 3. Put the spacing as text just above the rectangle
            text_position = (
                left,
                top - 10,
            )  # Slightly above the top-left corner
            cv2.putText(
                debug_img,
                f"A sidewalk",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # font scale
                (0, 255, 0),  # color = GREEN text
                2,  # thickness
            )
        self.debugger.save_image(debug_img, "most_probable_crosswalk.png")

        return most_probable_crosswalk, Minv, out.shape[:2]
