import torch
import cv2
import numpy as np
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg


from isaaclab.utils.noise import (
    AdditiveGaussianNoiseCfg as Gnoise
)

from witsense_ugv.envs.mdp import root_euler_xyz





# ///these are the commonly used observation terms 
@configclass
class BlindObsCfg:
    '''this is the default observation cconfigurations'''

    @configclass
    class PolicyCfg(ObsGroupCfg):
        '''observations for the policy group'''

        roos_pos_w_term =  ObsTermCfg(  #this is in meters
            func= mdp.root_pos_w,
            noise= Gnoise(mean=0.0, std= 0.1),
        )

        root_euler_xyz_term= ObsTermCfg(  # this is in radians
            func = root_euler_xyz,
            noise= Gnoise(mean=0.0, std= 0.1),
        )

        base_lin_vel_term= ObsTermCfg( #this is in m/s
            func= mdp.base_lin_vel,
            noise= Gnoise(mean=0.0, std= 0.5),
        )
        base_ang_vel_term= ObsTermCfg( #rad/s
            func= mdp.base_ang_vel,
            noise=Gnoise(std=0.4),
        )
        last_action_term= ObsTermCfg(
            func= mdp.last_action,
            clip=(-1., 1.)
        )

        def __post_init__(self):
            self.concatenate_terms= True
            self.enable_corruption= False

    policy: PolicyCfg = PolicyCfg()





