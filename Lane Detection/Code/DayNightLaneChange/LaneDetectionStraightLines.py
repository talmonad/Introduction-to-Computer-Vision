import cv2
from numpy.ma.core import clip
import matplotlib.pyplot as plt
from LaneChangeDetector import *
from moviepy.video.io.VideoFileClip import VideoFileClip


def preprocess_image(image):
    """
    Extract white lanes and perform edge detection.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for white
    lower_white = np.array([0, 0, 200])  # Adjust as needed
    upper_white = np.array([180, 30, 255])  # Adjust as needed

    # Mask white pixels
    mask = cv2.inRange(hsv, lower_white, upper_white)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    return blurred


def set_roi(edges):
    im_height, im_width = edges.shape[0], edges.shape[1]
    mask = np.zeros_like(edges)
    ignore_mask_color = [50, 50, 50]
    # bottom_left = (im_width*0.15, im_height)
    # bottom_right = (0.75*im_width, im_height)
    bottom_left = (0, im_height)
    bottom_right = (im_width, im_height)
    upper_left = (im_width*0.35, im_height*0.6)
    upper_right = (im_width*0.65, im_height * 0.6)
    vertices = np.array([[bottom_left, upper_left, upper_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges


def edge_detection(result, low_threshold=50, high_threshold=150):
    median_intensity = np.median(result)
    lower = int(max(0, 0.7 * median_intensity))
    upper = int(min(255, 1.3 * median_intensity))
    edges = cv2.Canny(result, lower, upper)
    return edges


def hough(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=100, maxLineGap=50)
    return lines


def post_process_line(lines):
    mxb = np.array([[0, 0]])
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2-x1 != 0:
                m = (y2 - y1) / (x2 - x1)
            else:
                m = -10000 if y2-y1 < 0 else 10000
            b = y1 + -1 * m * x1
            mxb = np.vstack((mxb, [m, b]))
    return mxb


def separate_left_right_lane_points(mxb):
    """
    Separate lane points into left and right lanes based on x-coordinate.
    """
    median_right_m = np.median(mxb[mxb[:, 0] > 0, 0])
    median_left_m = np.median(mxb[mxb[:, 0] < 0, 0])
    median_right_b = np.median(mxb[mxb[:, 0] > 0, 1])
    median_left_b = np.median(mxb[mxb[:, 0] < 0, 1])
    return median_left_m, median_right_m, median_left_b, median_right_b


def calculate_intersection_point(median_left_m, median_right_m, median_left_b, median_right_b, im_height):
    # Calculate the Intersect point of our two lines
    x_intersect = (median_left_b - median_right_b) / (median_right_m - median_left_m)
    y_intersect = median_right_m * (median_left_b - median_right_b) / (median_right_m - median_left_m) + median_right_b
    # Calculate the X-Intercept Points
    # x = (y - b) / m
    left_bottom = (im_height - median_left_b) / median_left_m
    right_bottom = (im_height - median_right_b) / median_right_m
    return x_intersect, y_intersect, left_bottom, right_bottom


def track_history_for_smoothing(x_intersect, y_intersect, left_bottom, right_bottom, history):
    # Create a History array for smoothing
    num_frames_to_median = 19
    new_history = [left_bottom, right_bottom, x_intersect, y_intersect]
    if (history.shape[0] == 1):  # First time, create larger array
        history = new_history
        for i in range(num_frames_to_median):
            history = np.vstack((history, new_history))
    elif (not (np.isnan(new_history).any())):
        history[:-1, :] = history[1:]
        history[-1, :] = new_history
    # Calculate the smoothed line points
    left_bottom_median = np.median(history[:, 0])
    right_bottom_median = np.median(history[:, 1])
    x_intersect_median = np.median(history[:, 2])
    y_intersect_median = np.median(history[:, 3])
    return left_bottom_median, right_bottom_median, x_intersect_median, y_intersect_median, history


def draw_lanes_on_image(image, left_bottom_median, right_bottom_median, x_intersect_median, y_intersect_median):
    im_height, im_width, im_depth = image.shape
    # Create a blank image to draw lines on
    line_image = np.copy(image) * 0
    # Create our Lines
    cv2.line(
        line_image,
        (np.int_(left_bottom_median), im_height),
        (np.int_(x_intersect_median), np.int_(y_intersect_median)),
        (255, 0, 0), 10
    )
    cv2.line(
        line_image,
        (np.int_(right_bottom_median), im_height),
        (np.int_(x_intersect_median), np.int_(y_intersect_median)),
        (0, 0, 255), 10
    )
    # Draw the lines on the image
    lane_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lane_edges


def lane_detection_pipeline(image, history):
    # Preprocess the image (HSV + Gaussian blur)
    preprocessed_image = preprocess_image(image)

    # Apply edge detection (Canny)
    edges = edge_detection(preprocessed_image)

    # Apply ROI mask to focus on the region of interest (ROI)
    roi_edges = set_roi(edges)

    lines = hough(roi_edges)
    if lines is None:
        return None, history, None
    mxb = post_process_line(lines)
    median_left_m, median_right_m, median_left_b, median_right_b = separate_left_right_lane_points(mxb)
    x_intersect, y_intersect, left_bottom, right_bottom = calculate_intersection_point(median_left_m, median_right_m, median_left_b, median_right_b, preprocessed_image.shape[0])
    left_bottom_median, right_bottom_median, x_intersect_median, y_intersect_median, history = track_history_for_smoothing(x_intersect, y_intersect, left_bottom, right_bottom, history)
    lane_edges = draw_lanes_on_image(image, left_bottom_median, right_bottom_median, x_intersect_median, y_intersect_median)
    lane_change = lane_change_detector.detect_lane_change(left_bottom_median, right_bottom_median)
    return lane_edges, history, lane_change


def process_video(input_filename, output_filename):
    history = np.array([[0, 0, 0, 0]])  # Initialize History
    images_list = []
    # Read in Base Video Clip
    base_clip = VideoFileClip(input_filename)

    for frame in base_clip.iter_frames(fps=30):  # Adjust fps as needed
        # Pass the frame through the lane detection pipeline
        lane_edges, history, lane_change = lane_detection_pipeline(frame, history)

        if lane_edges is not None:
            # Display the lane change status
            if lane_change:
                cv2.putText(lane_edges, f"Lane Change: {lane_change}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
            # Convert the processed frame (RGB) to BGR for OpenCV display
            lane_edges_bgr = cv2.cvtColor(lane_edges, cv2.COLOR_RGB2BGR)

            # Display the frame live
            cv2.imshow('Lane Detection', lane_edges_bgr)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release any resources and close windows
    cv2.destroyAllWindows()

    save_processed_video(input_filename, output_filename)





def save_processed_video(input_filename, output_filename):
    history = np.array([[0, 0, 0, 0]])  # Initialize History

    # Read input video
    base_clip = VideoFileClip(input_filename)

    # Get video properties
    frame_width, frame_height = base_clip.size
    fps = base_clip.fps

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI file
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Process each video frame
    for frame in base_clip.iter_frames(fps=fps):
        # Pass the frame through the lane detection pipeline
        lane_edges, history, lane_change = lane_detection_pipeline(frame, history)

        if lane_edges is not None:
            if lane_change:
                cv2.putText(lane_edges, f"Lane Change: {lane_change}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
            # Convert the processed frame (RGB) to BGR for OpenCV saving
            lane_edges_bgr = cv2.cvtColor(lane_edges, cv2.COLOR_RGB2BGR)

            # Write the frame to the output video
            out.write(lane_edges_bgr)

    # Release the VideoWriter
    out.release()

    print(f"Video saved to {output_filename}")



if __name__ == "__main__":
    # Example usage
    input_video_path = 'Lane Detection.mp4'  # Path to your input video
    output_video_path = 'output_video.mp4'  # Path to save the output video
    # Initialize the lane change detector
    lane_change_detector = LaneChangeDetector(threshold=25)

    # Process the video
    #process_video(input_video_path, output_video_path)
    process_video(input_video_path, output_video_path)