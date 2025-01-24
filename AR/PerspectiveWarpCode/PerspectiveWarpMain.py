from ImageWarper import *
from Visualizer import *
from ImageProcessor import *
from VideoProcessor import *
from FeatureMatcher import *
from PipelineController import *

if __name__ == "__main__":
    TEMPLATE_IMAGE_PATH = 'roee.jpeg'
    VIDEO_INPUT_PATH = 'roee.mp4'
    VIDEO_OUTPUT_PATH = 'path_to_output_video.mp4'
    OUTPUT_IMAGE_DIRECTORY = 'output_images'

    # class declaration
    visualizer = None
    #visualizer = Visualizer()
    image_warper = ImageWarper()
    template_processor = ImageProcessor(TEMPLATE_IMAGE_PATH, visualizer)
    video_processor = VideoProcessor(VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH, OUTPUT_IMAGE_DIRECTORY)
    template_processor.process_template()
    feature_matcher = FeatureMatcher(template_processor.keypoints, template_processor.descriptors)

    # pipeline
    pipeline = PipelineController(template_processor, video_processor, feature_matcher, visualizer, image_warper)
    pipeline.process_video()