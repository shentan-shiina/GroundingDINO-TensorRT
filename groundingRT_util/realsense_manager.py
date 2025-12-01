import pyrealsense2 as rs

class RealSenseManager:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(config)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        return aligned.get_depth_frame(), aligned.get_color_frame()

    def stop(self):
        self.pipeline.stop()


class RealSenseManager_v2:
    def __init__(self,input_config):
        self.pipeline = rs.pipeline()
        config = rs.config()
        w, h = input_config['cam_resolution']
        fps = input_config['fps']
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(config)
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        return aligned.get_depth_frame(), aligned.get_color_frame()

    def stop(self):
        self.pipeline.stop()