import numpy as np
import cv2 as cv
import glob


class CalibrateCamera:
    def __init__(self):
        pass

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

            # If found, add object points, image points (after refining them)
            if ret:
                print(f"Finished successfully {fname}!")
                objpoints.append(objp)

                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                #cv.drawChessboardCorners(img, pattern_size, corners2, ret)
                # resized_img = cv.resize(img, (800, 600))
                # cv.imshow('img', resized_img)
                # cv.waitKey(500)
            else:
                print(f"Could not find corners for {fname}!")
        #cv.destroyAllWindows()
        return objpoints, imgpoints, img_shape

    def calibrate(self, objpoints, imgpoints, shape):
        rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, None, None)
        newcameramtx, roi = self.get_optimal_camera_matrix(mtx, dist, shape)
        print("\nRMS:", rms)
        print("camera matrix:\n", mtx)
        print("distortion coefficients: ", dist.ravel())
        # Save the camera matrix, distortion coefficients, optimal camera matrix, and ROI
        np.savez("camera_calibration_data.npz", mtx=mtx, dist=dist, newcameramtx=newcameramtx, roi=roi)
        print("Calibration data saved to camera_calibration_data.npz")



    def get_optimal_camera_matrix(self, mtx, dist, shape):
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, shape, 0.5, shape)
        return newcameramtx, roi



if __name__ == '__main__':
    cb = CalibrateCamera()
    objpoints, imgpoints, img_shape = cb.findCorners()
    h, w = img_shape
    cb.calibrate(objpoints, imgpoints, (w, h))