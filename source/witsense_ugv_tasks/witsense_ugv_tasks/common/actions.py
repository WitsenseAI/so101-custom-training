from isaaclab.utils import configclass
from witsense_ugv.envs.mdp import RCCar4WDActionCfg


@configclass
class Mushr4WDActionCfg:

    throttle_steer = RCCar4WDActionCfg(
        wheel_joint_names= [
            "back_left_wheel_throttle",
            "back_right_wheel_throttle",
            "front_left_wheel_throttle",
            "front_right_wheeel_throttle",
        ],
        steering_joint_names= [
            "front_left_wheel_steer",
            "front_right_wheel_steer",

        ],
        base_length= 0.325,
        base_width= 0.2,
        wheel_radious= 0.05,
        scale= (3.0, 0.488),
        no_reverse= True,
        bounding_strategy= "clip",
        assest_name= "robot",

    )
