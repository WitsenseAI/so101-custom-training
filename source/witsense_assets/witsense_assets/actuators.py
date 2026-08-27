from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg


#actuators config fo rthe mushr_nano robot
MUSHR_ACTUATOR_CFG={
    "steering_joints":ImplicitActuatorCfg(
        joint_names_expr= ['front_left_wheel_steer', "front_right_wheel_steer"],
        velocity_limit = 10.0,
        effort_limit= 3.2,
        stiffness= 100.0,
        damping= 10.0,
        friction= 0.0,
    ),
    "throttle_joints": DCMotorCfg(
        joint_names_expr=[".*throttle"],
        saturation_effort = 1.05,
        effort_limit= 0.25,
        velocity_limit= 450.0,
        stiffness= 0.0,
        damping= 1000.0,
        friction= 0.0,
    ),
}

MUSHR_SUS_ACTUATORS_CFG = { #thsi is for the 4wd drive
    **MUSHR_ACTUATOR_CFG,
    "suspension": ImplicitActuatorCfg (
        joint_names_expr= [".*_suspension"],
        effort_limit= None,
        velocity_limit= None,
        stiffness= 1e8,
        damping =  0.,
        friction= 0.5,


    )


}
