import numpy as np
import cv2


class PerspectiveWarp:
    def __init__(self, dst_size=(1920, 1080),
                 src=np.float32([(0.45, 0.65), (0.55, 0.65), (0.1, 1.0), (0.9, 1.0)]),
                 dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
        """
        Initialize the PerspectiveWarp class with default source and destination points.
        :param dst_size: Size of the destination image (width, height).
        :param src: Source points as a percentage of the original image size.
        :param dst: Destination points as a percentage of the destination image size.
        """
        self.dst_size = dst_size
        self.src = src
        self.dst = dst

    def perspective_warp(self, img):
        """
        Apply a perspective warp to the input image.
        :param img: Input image to be warped.
        :return: Warped image with a bird's eye view.
        """
        if img is None or len(img.shape) < 2:
            raise ValueError("Invalid input image. Make sure it's a valid 2D or 3D array.")

        img_size = np.float32([img.shape[1], img.shape[0]])  # Width, Height
        src_scaled = self.src * img_size
        dst_scaled = self.dst * np.float32(self.dst_size)

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_scaled, dst_scaled)

        # Perform the perspective warp
        warped = cv2.warpPerspective(img, M, self.dst_size)
        return warped

    def visualize_points(self, img):
        """
        Visualize the source and destination points on the input image.
        :param img: Input image for visualization.
        :return: Image with source points overlaid.
        """
        if img is None or len(img.shape) < 2:
            raise ValueError("Invalid input image. Make sure it's a valid 2D or 3D array.")
        src = self.src
        img_size = np.float32([img.shape[1], img.shape[0]])  # Width, Height
        src_scaled = (src * img_size).astype(int)

        # Overlay the source points on the image
        img_copy = img.copy()
        for point in src_scaled:
            cv2.circle(img_copy, tuple(point), 10, (0, 255, 0), -1)

        return img_copy

    def set_src_points(self, new_src):
        """
        Update the source points.
        :param new_src: New source points as a percentage of the image size.
        """
        if len(new_src) != 4:
            raise ValueError("Source points must contain exactly 4 points.")
        self.src = np.float32(new_src)

    def set_dst_points(self, new_dst):
        """
        Update the destination points.
        :param new_dst: New destination points as a percentage of the destination size.
        """
        if len(new_dst) != 4:
            raise ValueError("Destination points must contain exactly 4 points.")
        self.dst = np.float32(new_dst)

    def inv_perspective_warp(self, img):
        """
        Apply the inverse perspective warp to the input image.
        :param img: Input image to be inversely warped.
        :return: Image warped back to the original perspective.
        """
        if img is None or len(img.shape) < 2:
            raise ValueError("Invalid input image. Make sure it's a valid 2D or 3D array.")

        img_size = np.float32([img.shape[1], img.shape[0]])  # Width, Height
        src_scaled = self.dst * np.float32(self.dst_size)
        dst_scaled = self.src * img_size

        # Compute the inverse perspective transform matrix
        M = cv2.getPerspectiveTransform(src_scaled, dst_scaled)

        # Perform the inverse perspective warp
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        return warped
