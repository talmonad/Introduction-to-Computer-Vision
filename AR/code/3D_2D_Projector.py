from ImageWarper import *
from Visualizer import *
from ImageProcessor import *
from VideoProcessor import *
from FeatureMatcher import *
from PipelineController import *
from CalibrateCamera import *
from Processor3D import *

if __name__ == "__main__":
    TEMPLATE_IMAGE_PATH = 'trevi_port.jpg'
    VIDEO_INPUT_PATH = 'trevi_port.MOV'
    VIDEO_OUTPUT_PATH = 'path_to_output_video.mp4'
    OUTPUT_IMAGE_DIRECTORY = 'output_images'

    # class declaration
    visualizer = None
    visualizer = Visualizer()
    image_warper = ImageWarper()
    cb = CalibrateCamera()
    p3d = Processor3D("rabbit.obj")
    #p3d = Processor3D("Dragon 2.5_ply.ply")
    template_processor = ImageProcessor(TEMPLATE_IMAGE_PATH, visualizer, cb)
    video_processor = VideoProcessor(VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH, OUTPUT_IMAGE_DIRECTORY)
    template_processor.process_template(calibrate=True)
    feature_matcher = FeatureMatcher(template_processor.keypoints, template_processor.descriptors)

    # pipeline
    pipeline = PipelineController(template_processor, video_processor, feature_matcher, visualizer, image_warper, cb, p3d)
    pipeline.process_video(project_cube=True, calibrate=True)