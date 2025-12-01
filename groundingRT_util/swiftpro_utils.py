from uarm.wrapper import SwiftAPI
import numpy as np
class SwiftPro:
    """SwiftPro API"""

    def __init__(self):
        self.swift = SwiftAPI()
        self.swift.waiting_ready()
        self.swift.flush_cmd()
        self.swift.connect()
        self.swift.set_polar(stretch=150, rotation=90, height=100, speed=20)
        self.polar = [0, 90, 0]  # [stretch, rotation, height]

    def get_polar(self):
        return self.swift.get_polar()

    def set_polar(self, stretch=None, rotation=None, height=None, speed = 100):
        if stretch is not None:
            stretch = np.min([stretch, 299])
        if rotation is not None:
            rotation = np.min([rotation, 135])
        if height is not None:
            height = np.min([height, 150])
        self.swift.set_polar(stretch=stretch, rotation=rotation, height=height, speed=speed)
        return self.get_polar()

    def set_gripper(self,state=False):
        return self.swift.set_gripper(state)