import cv2
import numpy as np
from numpy.ma.core import clip
import matplotlib.pyplot as plt
from LaneChangeDetector import *
from moviepy.video.io.VideoFileClip import VideoFileClip
from CameraCalibration import *
from PerspectiveWarp import *
from LaneSeparator import *
from CurveFitDetector import *
from CrosswalkDetector import CrosswalkDetector
from CrosswalkDetectionState import CrosswalkDetectionState


def night_time_processing(img, gamma=0.4):
    """
    Preprocess image for lane detection with adjustments for night mode.
    """
    # Apply gamma correction
    gamma_corrected = np.power(img / 255.0, gamma)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    return gamma_corrected


def edge_detection(result):
    median_intensity = np.median(result)
    lower = int(max(0, 0.7 * median_intensity))
    upper = int(min(255, 1.3 * median_intensity))
    edges = cv2.Canny(result, lower, upper)
    return edges


def preprocess_image_night(image, night):
    low_h, low_s, low_v, high_h, high_s, high_v = 0, 0, 200, 180, 255, 255
    block_size = 15
    kernel_size = 5
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    if not night:
        shadow_mask = cv2.inRange(l_channel, 0, 70)
        l_channel[shadow_mask > 0] = 0
        block_size = 3
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    # Merge CLAHE-enhanced L-channel back with a and b channels
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    # Convert to HSV color space
    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)

    # Define HSV range for white
    lower_white = np.array([low_h, low_s, low_v])  # Adjust as needed
    upper_white = np.array([high_h, high_s, high_v])  # Adjust as needed

    # Mask white pixels
    mask = cv2.inRange(hsv, lower_white, upper_white)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        2,
    )
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(adaptive_thresh, (kernel_size, kernel_size), 0)
    edges = edge_detection(blurred)
    return edges


def preprocess_image_day(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    """
    Preprocess the image to extract lane edges using Sobel gradients and color thresholds.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    shadow_mask = cv2.inRange(
        l_channel, 0, 70
    )  # Adjust threshold for shadow detection
    l_channel[shadow_mask > 0] = 0
    # Assuming l_channel is defined and is a grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Ensure l_channel is in the correct data type
    l_channel_uint8 = l_channel.astype(np.uint8)

    # Apply CLAHE to the entire channel
    enhanced_l_channel = clahe.apply(l_channel_uint8)
    # Sobel x
    sobelx = cv2.Sobel(enhanced_l_channel, cv2.CV_64F, 1, 1)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[
        (scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])
    ] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def pre_process_image_cw(img, night=False):
    """
    Preprocess the image with shadow removal and CLAHE.

    Parameters:
        image (ndarray): Input BGR image.

    Returns:
        ndarray: Preprocessed grayscale image.
    """
    # Step 1: Shadow removal

    # Step 2: Convert to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Smooth the image to remove noise
    kernel_size = 5
    blur_img = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    # Step 4: Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(5, 5))
    clahe_image = clahe.apply(blur_img)

    return clahe_image


def crosswalk_detection_pipeline(frame, cb, cw, night=False):
    """
    Complete pipeline for crosswalk detection on a single frame.
    """
    # Cross walk detection pipeline
    calib_im = cb.undistort(frame)
    pre_processed_image = pre_process_image_cw(frame, night)
    most_probable_crosswalk, cw_Minv, cw_warped_shape = cw.detect_crosswalk(
        frame, pre_processed_image
    )
    is_cw_detected = most_probable_crosswalk is not None
    return most_probable_crosswalk, is_cw_detected, cw_Minv, cw_warped_shape


def lane_detection_pipeline(image, cb, pw, ls, cf, lcd, night):
    """
    Complete pipeline for lane detection on a single frame.
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    calib_img = cb.undistort(img)
    edges = preprocess_image_night(calib_img, night)
    # edges = preprocess_image_night(calib_img) if night else preprocess_image_day(calib_img)
    # plt.imshow(pw.visualize_points(img))
    # plt.show()
    roi_edges = pw.perspective_warp(edges)
    out_img, curves, lanes, ploty = ls.sliding_window(
        roi_edges, draw_windows=False
    )
    out_img = cf.draw_lanes(img, curves[0], curves[1], perspective_warp=pw)
    lane_change = lcd.detect_lane_change(curves[0], curves[1])
    return out_img, lane_change


def compute_temporal_median(frame, median_history, max_frames=50):
    """
    Compute the temporal median from a sequence of frames.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median_history.append(gray_frame)

    if len(median_history) > max_frames:
        median_history.pop(0)  # Maintain a fixed window size

    return median_history


def initialize_classes():
    """
    Initialize and return required objects for the pipeline.
    """
    cb = CameraCalibration()
    src_night = np.float32(
        [(0.26, 0.70), (0.53, 0.70), (0.0, 1.0), (1.0, 1.0)]
    )
    src_day = np.float32([(0.4, 0.8), (0.58, 0.8), (0.2, 1.0), (0.8, 1.0)])
    cw_src = np.float32([(0.4, 0.8), (0.58, 0.8), (0.2, 1.0), (0.8, 1.0)])
    pw_cw_day = PerspectiveWarp(src=cw_src)
    pw_night = PerspectiveWarp(src=src_night)
    pw_day = PerspectiveWarp(src=src_day)

    ls = LaneSeparator()
    cf = CurveFit()
    lcd = LaneChangeDetector(threshold=375, temporal_window=100)
    cw = CrosswalkDetector()
    cw_det_state = CrosswalkDetectionState()
    return cb, pw_night, pw_day, ls, cf, lcd, cw, cw_det_state, pw_cw_day


def plot_temporal_median_histogram(temporal_median, median_history):
    """
    Plot a histogram of the grayscale intensity values of the temporal median.

    Parameters:
    - median_history: List of grayscale frames used for temporal median computation.
    """
    if len(median_history) == 0:
        print("No frames in the median history to plot.")
        return

    # Flatten the temporal median for histogram plotting
    flattened_median = temporal_median.flatten()

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(
        flattened_median,
        bins=256,
        range=(0, 255),
        color="blue",
        alpha=0.7,
        label="Temporal Median",
    )
    plt.title("Histogram of Temporal Median Grayscale Intensity")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def display_lane_change_status(
    lcd,
    lane_change,
    lane_edges,
    lane_change_count,
    keep_printing,
    last_lane_change,
):
    if lane_change:
        if sum(list(lcd.temporal_lane_change_flags)[-10:]) >= 5:
            last_lane_change = lane_change
            cv2.putText(
                lane_edges,
                f"Lane Change: {lane_change}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            keep_printing = True
    if keep_printing:
        cv2.putText(
            lane_edges,
            f"Lane Change: {last_lane_change}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        lane_change_count -= 1
        if lane_change_count == 0:
            lane_change_count = 18
            keep_printing = False
    return lane_change_count, keep_printing, last_lane_change


def put_text_on_frame(
    image, input_text, col_place=0.1, row_place=0.1, color=(255, 255, 255)
):
    overlay = image.copy()
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    text_size = cv2.getTextSize(input_text, font, scale, thickness)[0]
    text_x = int(width * col_place)
    text_y = int(height * row_place)

    # Draw a semi-transparent rectangle
    box_coords = (
        (text_x - 10, text_y - 20),
        (text_x + text_size[0] + 10, text_y + text_size[1]),
    )
    cv2.rectangle(
        overlay, box_coords[0], box_coords[1], (0, 0, 0), -1
    )  # Black box

    # Blend the rectangle with the image
    overlayed_image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    # Draw the text
    cv2.putText(
        overlayed_image,
        input_text,
        (text_x, text_y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return overlayed_image


def overlay_crosswalk_on_original(
    original_frame: np.ndarray,
    warped_shape: tuple,
    rectangle: tuple,
    Minv: np.ndarray,
    color=(255, 0, 0),  # Default: blue for crosswalk
    overlay_alpha=0.3,
):
    """
    Overlay the detected crosswalk rectangle onto the original camera view.

    Parameters:
        original_frame (ndarray): The original video frame (BGR).
        warped_shape (tuple): Shape of the warped image (height, width).
        rectangle (tuple): Coordinates of the rectangle (left, right, top, bottom) in the warped image.
        Minv (ndarray): Inverse perspective transformation matrix (from warp to original view).
        color (tuple): BGR color for the crosswalk rectangle (default: blue).
        overlay_alpha (float): Transparency factor for blending the rectangle with the original frame.

    Returns:
        overlayed_frame (ndarray): The original frame with the crosswalk rectangle overlaid.
    """
    if rectangle is None:
        # No valid rectangle detected
        return original_frame

    # Extract rectangle coordinates
    left, right, top, bottom = rectangle

    # Create a blank mask in the warped space
    crosswalk_mask_warped = np.zeros(
        (warped_shape[0], warped_shape[1], 3), dtype=np.uint8
    )

    # Create the rectangle points
    rect_points = np.array(
        [
            [left, top],  # Top-left corner
            [right, top],  # Top-right corner
            [right, bottom],  # Bottom-right corner
            [left, bottom],  # Bottom-left corner
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    # Draw the filled rectangle in the warped space
    cv2.fillPoly(crosswalk_mask_warped, [rect_points], color)

    # Unwarp the crosswalk mask back to the original camera view
    unwarped_crosswalk_mask = cv2.warpPerspective(
        crosswalk_mask_warped,
        Minv,
        (original_frame.shape[1], original_frame.shape[0]),
    )

    # Blend the unwarped rectangle with the original frame
    overlayed_frame = cv2.addWeighted(
        original_frame, 1.0, unwarped_crosswalk_mask, overlay_alpha, 0
    )

    return overlayed_frame


def process_video_live(input_filename):
    """
    Process a video live for lane detection and display results with temporal median analysis.
    """
    base_clip = VideoFileClip(input_filename)
    cb, pw_night, pw_day, ls, cf, lcd, cw, cw_det_state, pw_cw_day = (
        initialize_classes()
    )

    count = 0
    lane_change_count = 18
    keep_printing = False
    last_lane_change = None
    median_history = []

    for frame in base_clip.iter_frames(fps=10):
        # Compute the temporal median
        median_history = compute_temporal_median(
            frame, median_history, max_frames=20
        )
        temporal_median = np.median(np.stack(median_history), axis=0).astype(
            np.uint8
        )
        mean_brightness = np.mean(temporal_median)

        pw = pw_night if mean_brightness < 70 else pw_day
        night = True if mean_brightness < 70 else False

        # Pass the frame throught the crosswalk detection pipeline first
        most_probable_crosswalk, is_cw_detected, cw_Minv, cw_warped_shape = (
            crosswalk_detection_pipeline(frame, cb, cw, night)
        )
        cw_det_state.update_window(is_cw_detected)
        copy_frame = frame.copy()
        if cw_det_state.should_signal_text():
            copy_frame = put_text_on_frame(
                copy_frame,
                "Crosswalk Detected! Slow Down!",
                row_place=0.1,
                col_place=0.7,
                color=(255, 0, 0),
            )
        cw_overlayed_frame = copy_frame
        if cw_det_state.should_display_frame():
            smoothed_cw = cw_det_state.smooth_crosswalk_identifier(
                most_probable_crosswalk
            )
            cw_overlayed_frame = overlay_crosswalk_on_original(
                copy_frame,
                cw_warped_shape,
                smoothed_cw,
                cw_Minv,
                overlay_alpha=0.6,
            )

        # Pass the frame through the lane detection pipeline
        lane_edges, lane_change = lane_detection_pipeline(
            cw_overlayed_frame, cb, pw, ls, cf, lcd, night
        )
        if lane_edges is not None:
            # Display the lane change status
            lane_change_count, keep_printing, last_lane_change = (
                display_lane_change_status(
                    lcd,
                    lane_change,
                    lane_edges,
                    lane_change_count,
                    keep_printing,
                    last_lane_change,
                )
            )
            # Convert the processed frame (RGB) to BGR for OpenCV display
            lane_edges_bgr = cv2.cvtColor(lane_edges, cv2.COLOR_RGB2BGR)

            # Display the frame live
            cv2.imshow("Lane Detection", lane_edges_bgr)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        count += 1
        # Plot the temporal histogram periodically
        if count % 100 == 0:
            # lcd.plot_temporal_hist()
            # temporal_median = np.median(np.stack(median_history), axis=0).astype(np.uint8)
            # plot_temporal_median_histogram(temporal_median, median_history)
            plt.imshow(pw.visualize_points(lane_edges))
            plt.show()
            pass
    # Release any resources and close windows
    cv2.destroyAllWindows()


def save_processed_video(input_filename, output_filename):
    history = np.array([[0, 0, 0, 0]])  # Initialize History

    # Read input video
    base_clip = VideoFileClip(input_filename)

    # Get video properties
    frame_width, frame_height = base_clip.size
    fps = base_clip.fps

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for AVI file
    out = cv2.VideoWriter(
        output_filename, fourcc, fps, (frame_width, frame_height)
    )
    cb, pw_night, pw_day, ls, cf, lcd, cw, cw_det_state = initialize_classes()
    median_history = []
    lane_change_count = 18
    keep_printing = False
    last_lane_change = None
    # Process each video frame
    for frame in base_clip.iter_frames(fps=fps):
        # Pass the frame through the lane detection pipeline
        median_history = compute_temporal_median(
            frame, median_history, max_frames=20
        )
        temporal_median = np.median(np.stack(median_history), axis=0).astype(
            np.uint8
        )
        mean_brightness = np.mean(temporal_median)
        pw = pw_night if mean_brightness < 0 else pw_day
        night = True if mean_brightness < 0 else False

        # Pass the frame throught the crosswalk detection pipeline first
        most_probable_crosswalk, is_cw_detected, cw_Minv, cw_warped_shape = (
            crosswalk_detection_pipeline(frame, cb, cw, night)
        )
        cw_det_state.update_window(is_cw_detected)
        copy_frame = frame.copy()
        if cw_det_state.should_signal_text():
            copy_frame = put_text_on_frame(
                copy_frame,
                "Crosswalk Detected! Slow Down!",
                row_place=0.1,
                col_place=0.7,
                color=(255, 0, 0),
            )
        cw_overlayed_frame = copy_frame
        if cw_det_state.should_display_frame():
            smoothed_cw = cw_det_state.smooth_crosswalk_identifier(
                most_probable_crosswalk
            )
            cw_overlayed_frame = overlay_crosswalk_on_original(
                copy_frame,
                cw_warped_shape,
                smoothed_cw,
                cw_Minv,
                overlay_alpha=0.6,
                color=(0, 0, 255),
            )
        # Pass the frame through the lane detection pipeline
        lane_edges, lane_change = lane_detection_pipeline(
            cw_overlayed_frame, cb, pw, ls, cf, lcd, night
        )
        if lane_edges is not None:
            lane_change_count, keep_printing, last_lane_change = (
                display_lane_change_status(
                    lcd,
                    lane_change,
                    lane_edges,
                    lane_change_count,
                    keep_printing,
                    last_lane_change,
                )
            )
            # Convert the processed frame (RGB) to BGR for OpenCV saving
            lane_edges_bgr = cv2.cvtColor(lane_edges, cv2.COLOR_RGB2BGR)

            # Write the frame to the output video
            out.write(lane_edges_bgr)
    # Release the VideoWriter
    out.release()

    print(f"Video saved to {output_filename}")


if __name__ == "__main__":
    # Example usage
    # input_video_path = 'day_and_night_cut.mp4'  # Path to your input video
    # input_video_path = "day_and_night.mp4"
    # Test video on crosswalk detection input video
    input_video_path = "./CrossWalkE.mp4"
    output_video_path = "./output_video.mp4"  # Path to save the output video

    # Process the video
    process_video_live(input_video_path)
    # save_processed_video(input_video_path, output_video_path)
