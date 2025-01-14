from collections import deque


class CrosswalkDetectionState:
    def __init__(self, frame_window=6, alpha=0.8):
        """Set a cross walk detection mechanism"""
        self.frame_window = frame_window
        self.found_hist = deque([])
        self.alpha = alpha  # Smoothing factor for identifier
        self.prev_box = None  # Store the previous bounding box for smoothing
        self.active_counter = 0

    def should_display_frame(self):
        """
        Updates the current window and determines if the frame should be displayed.

        Returns:
            bool: True if the frame should be displayed, False otherwise.
        """
        if len(self.found_hist) < 2:
            return False  # Not enough data yet

        valid_detections = sum(self.found_hist)
        last_two_valid = list(self.found_hist)[-2:] == [True, True]

        should_display = valid_detections >= 3 and last_two_valid
        if should_display:
            self.active_counter = 8
        return should_display

    def should_signal_text(self):
        """
        Determine if a text signal should be displayed.
        Returns:
            bool: True if a text signal is active, False otherwise.
        """
        signal = self.active_counter > 0
        self.active_counter -= 1
        # Check if there is at least one TRUE in the last `frame_window` frames
        return signal

    def update_window(self, detected=False):
        self.found_hist.append(detected)
        if len(self.found_hist) > self.frame_window:
            self.found_hist.popleft()

    def smooth_crosswalk_identifier(self, cur_box):
        """
        Smooth the crosswalk identifier position using weighted interpolation.
        """
        if self.prev_box is None:
            self.prev_box = cur_box  # Initialize with the first box
            return cur_box

        # Interpolate between the current and previous bounding boxes
        smoothed_box = tuple(
            int(self.alpha * cur + (1 - self.alpha) * prev)
            for cur, prev in zip(cur_box, self.prev_box)
        )
        self.prev_box = smoothed_box  # Update the previous box
        return smoothed_box
