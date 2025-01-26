import cv2


class VideoProcessor:
    def __init__(self, video_input_path, video_output_path, output_image_dir):
        self.video_input_path = video_input_path
        self.video_output_path = video_output_path
        self.output_image_dir = output_image_dir
        self.cap = None
        self.out = None

    def initialize(self):
        self.cap = cv2.VideoCapture(self.video_input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(self.video_output_path, fourcc, 20.0, (width, height))

    def release(self):
        self.cap.release()
        self.out.release()