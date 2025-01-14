import os
import cv2


class Debugger:
    def __init__(self, debug_mode, output_dir="debug_outputs"):
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        if debug_mode:
            os.makedirs(output_dir, exist_ok=True)

    def save_image(self, image, filename, readonly=False):
        if self.debug_mode:
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, image.copy() if readonly else image)
