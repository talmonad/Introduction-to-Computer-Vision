import cv2
import numpy as np

# Constants
TEMPLATE_IMAGE_PATH = 'camera_cal\\IMG_1772.jpg'
VIDEO_INPUT_PATH = 'checkerboard4k.MOV'
VIDEO_OUTPUT_PATH = 'path_to_output_video.mp4'

# Load calibration data
calibration_data = np.load('camera_calibration_data.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']
newcameramtx = calibration_data['newcameramtx']
roi = calibration_data['roi']

# Load and undistort template image
template_img = cv2.imread(TEMPLATE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
template_img = cv2.undistort(template_img, mtx, dist, None)
h, w = template_img.shape
# Find keypoints and descriptors in the undistorted template image
sift = cv2.SIFT_create()
template_kp, template_des = sift.detectAndCompute(template_img, None)

# Open video input and prepare video output
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Feature matcher
bf = cv2.BFMatcher()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (w, h))

    # Undistort the frame
    frame = cv2.undistort(frame, mtx, dist, None)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_kp, frame_des = sift.detectAndCompute(gray_frame, None)

    # Find keypoints matches
    matches = bf.knnMatch(template_des, frame_des, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > 10:
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp template image onto the frame
        h, w = template_img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # Plot and save frame
    out.write(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()