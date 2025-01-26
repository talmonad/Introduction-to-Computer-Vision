import os
import numpy as np
import cv2 as cv
import glob
from MeshRenderer import *
from matplotlib import pyplot as plt


class CalibrateCamera:
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        if os.path.exists('camera_calibration_data.npz'):
            d = np.load('camera_calibration_data.npz')
            self.mtx, self.dist, self.rvecs, self.tvecs = d['mtx'], d['dist'], d['rvecs'], d['tvecs']

    def get_images(self):
        return glob.glob("camera_cal\\" + "*.jpg")

    def findCorners(self):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        pattern_size = (9, 6)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((np.prod(pattern_size), 3), np.float32)
        objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = self.get_images()

        for fname in images:
            print(f"Processing {fname}...")
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_shape = gray.shape
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
            imgpoints.append(corners.reshape(-1, 2))
            objpoints.append(objp)
            # If found, add object points, image points (after refining them)
            # if ret:
            #     print(f"Finished successfully {fname}!")
            #     objpoints.append(objp)
            #
            #     corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            #     imgpoints.append(corners2)
            #
            #     # Draw and display the corners
            #     cv.drawChessboardCorners(img, pattern_size, corners2, ret)
            #     resized_img = cv.resize(img, (800, 600))
            #     cv.imshow('img', resized_img)
            #     cv.waitKey(500)
            # else:
            #     print(f"Could not find corners for {fname}!")
        #cv.destroyAllWindows()
        return objpoints, imgpoints, img_shape

    def calibrate(self, objpoints, imgpoints, shape):
        rms, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, None, None)
        print("\nRMS:", rms)
        print("camera matrix:\n", self.mtx)
        print("distortion coefficients: ", self.dist.ravel())
        # Save the camera matrix, distortion coefficients, optimal camera matrix, and ROI
        np.savez("camera_calibration_data.npz", mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
        print("Calibration data saved to camera_calibration_data.npz")


    def undistort_image(self, img):
        return cv.undistort(img, self.mtx, self.dist, None, self.mtx)

    def get_optimal_camera_matrix(self, shape):
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, shape, 0.5, shape)
        return newcameramtx, roi

    def draw(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # draw ground floor in green
        img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

        # draw top layer in red color
        img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

        return img

    def draw_cube(self):
        square_size = 1.0
        objectPoints = (
                3
                * square_size
                * np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, -1],
                [1, 1, -1],
                [1, 0, -1],
            ]
        )
        )



        figsize = (20, 20)
        plt.figure(figsize=figsize)
        img_names = self.get_images()
        for i, fn in enumerate(img_names):
            imgBGR = cv.imread(fn)
            imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB)
            mr = MeshRenderer(self.mtx, imgRGB.shape[1], imgRGB.shape[0], "rabbit.obj")
            out_frame = mr.draw(imgRGB, self.rvecs[i], self.tvecs[i])
            imgpts = cv.projectPoints(objectPoints, self.rvecs[i], self.tvecs[i], self.mtx, self.dist)[0]
            drawn_image = self.draw(imgRGB, imgpts)

            if i < 12:
                #plt.subplot(4, 3, i + 1)
                plt.imshow(drawn_image)
                plt.show()

        plt.show()

    def adjust_camera_matrix(self, K, W_cal, H_cal, W_use, H_use):
        """Adjusts the camera matrix for a different resolution."""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        fx_new = fx * (W_use / W_cal)
        fy_new = fy * (H_use / H_cal)
        cx_new = cx * (W_use / W_cal)
        cy_new = cy * (H_use / H_cal)

        K_new = np.array([[fx_new, 0, cx_new],
                          [0, fy_new, cy_new],
                          [0, 0, 1]], dtype=np.float32)
        return K_new

if __name__ == '__main__':
    cb = CalibrateCamera()
    objpoints, imgpoints, img_shape = cb.findCorners()
    h, w = img_shape
    cb.calibrate(objpoints, imgpoints, (w, h))
    cb.draw_cube()