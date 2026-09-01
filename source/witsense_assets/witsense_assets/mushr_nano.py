

from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from .actuators import MUSHR_SUS_ACTUATORS_CFG

from . import MAIN_ASSETS_DIR

_ZERO_INIT_STATES  = ArticulationCfg.InitialStateCfg(
    pos= (0.0, 0.0, 0.0, 0.0),
    joint_pos ={
        'back_left_wheel_throttle' : 0.0,
        'back_right_wheel_throttle': 0.0,
        'front_left_wheel_steer': 0.0,
        'front_right_wheel_steer': 0.0,
        'front_left_wheel_throttle': 0.0,
        'front_right_wheel_throttle': 0.0,
        # '''from here the initial states for sustension joints starts'''
        'front_left_wheel_suspension': 0.0,
        'front_right_wheel_suspension': 0.0,
        'back_left_wheel_suspension': 0.0,
        'back_right_wheel_suspension': 0.0,
    },
)

MUSHR_SUS_CFG= ArticulationCfg(
    spawn= sim_utils.UsdFileCfg(
        usd_path= f"{MAIN_ASSETS_DIR}/robots/mushroom_nano/mushr_nano_v2.usd",
        rigid_props= sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled= True,
            max_linear_velocity= 1000.0,
            max_angular_velocity= 100000.0,
            max_depenetration_velocity= 100.0,
            max_contact_impulse= 0.0,
            enable_gyroscopic_forces= True,

        ),
        articulation_props= sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions= False,
            solver_position_iteration_count=4,
            # 0, not 4: upstream WheeledLab runs this asset at 0 for thousands of
            # iterations. With 4, the TGS solver hung inside PhysX after ~20 min of
            # wall clock (py-spy: blocked in physics_context._step, GPU pinned at 100%).
            solver_velocity_iteration_count= 0,
            sleep_threshold= 0.005,
            stabilization_threshold= 0.001,
        )
    ),
    init_state= _ZERO_INIT_STATES,
    actuators= MUSHR_SUS_ACTUATORS_CFG
    

)

