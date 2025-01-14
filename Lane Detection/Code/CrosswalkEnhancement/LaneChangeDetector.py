from collections import deque
import matplotlib.pyplot as plt
import numpy as np


class LaneChangeDetector:
    def __init__(self, threshold=100, temporal_window=50, history_limit=10):
        self.threshold = (
            threshold  # Sensitivity threshold for lane change detection
        )
        self.history_limit = history_limit  # Limit for historical data
        self.previous_left_bottom = deque(maxlen=history_limit)
        self.previous_right_bottom = deque(maxlen=history_limit)
        self.record_changes_left = []
        self.record_changes_right = []

        # Temporal buffers for tracking recent changes
        self.temporal_window = temporal_window
        self.temporal_left_changes = deque(maxlen=temporal_window)
        self.temporal_right_changes = deque(maxlen=temporal_window)
        self.temporal_lane_change_flags = deque(maxlen=temporal_window)

    def detect_lane_change(self, left_bottom_median, right_bottom_median):
        """
        Detect lane changes by comparing the bottom positions of the lanes.
        """
        if left_bottom_median is None or right_bottom_median is None:
            return None  # Handle missing data gracefully

        lane_change = None  # Default: No lane change

        # Process right lane change
        if self.previous_right_bottom and self.previous_left_bottom:
            prev_right = np.mean(self.previous_right_bottom, axis=0)
            prev_left = np.mean(self.previous_left_bottom, axis=0)
            right_change = np.mean(right_bottom_median - prev_right)
            left_change = np.mean(left_bottom_median - prev_left)
            self.record_changes_right.append(right_change)
            self.temporal_right_changes.append(right_change)
            self.record_changes_left.append(left_change)
            self.temporal_left_changes.append(left_change)

            if left_change > self.threshold:
                lane_change = "Right"
            elif right_change < -1 * self.threshold:
                lane_change = "Left"

        # Track lane change events
        self.temporal_lane_change_flags.append(1 if lane_change else 0)

        # Update previous positions with a limited history
        self.previous_left_bottom.append(left_bottom_median)
        self.previous_right_bottom.append(right_bottom_median)

        return lane_change

    def smooth_changes(self, changes, smoothing_window=5):
        """
        Smooth changes using a moving average filter.
        """
        if len(changes) < smoothing_window:
            return np.array(changes)  # Not enough data to smooth
        return np.convolve(
            changes, np.ones(smoothing_window) / smoothing_window, mode="valid"
        )

    def plot_temporal_hist(self):
        """
        Plot temporal changes for left and right lanes over time, including lane change moments.
        """
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Smooth changes for better visualization
        smoothed_left = self.smooth_changes(list(self.temporal_left_changes))
        smoothed_right = self.smooth_changes(list(self.temporal_right_changes))

        # Plot left lane changes
        ax[0].plot(
            range(len(smoothed_left)), smoothed_left, label="Right Changes"
        )
        ax[0].axhline(
            self.threshold, color="r", linestyle="--", label="Threshold"
        )
        ax[0].set_title("Temporal Right Lane Changes")
        ax[0].set_ylabel("Change Magnitude")
        ax[0].legend()

        # Plot right lane changes
        ax[1].plot(
            range(len(smoothed_right)), smoothed_right, label="Left Changes"
        )
        ax[1].axhline(
            -1 * self.threshold, color="r", linestyle="--", label="Threshold"
        )
        ax[1].set_title("Temporal Left Lane Changes")
        ax[1].set_ylabel("Change Magnitude")
        ax[1].legend()

        # Plot lane change events
        ax[2].plot(
            range(len(self.temporal_lane_change_flags)),
            self.temporal_lane_change_flags,
            label="Lane Change Events",
        )
        ax[2].set_title("Lane Change Events Over Time")
        ax[2].set_ylabel("Change (0/1)")
        ax[2].set_yticks([0, 1])
        ax[2].set_yticklabels(["No Change", "Change"])
        ax[2].legend()

        plt.xlabel("Time (Frames)")
        plt.tight_layout()
        plt.show()
