import weakref
from collections.abc import Callable

import numpy as np

import carb
import omni

from ..device_base import Device


class Se3Gamepad(Device):
    """A gamepad controller for driving the SO-101's six joints directly.

    Joint-space, not task-space: this produces the same per-joint deltas as
    ``Se3Keyboard`` so it drops into ``preprocess_device_action``'s ``keyboard``
    branch unchanged. Isaac Lab ships an ``Se3Gamepad`` too, but that one emits SE(3)
    delta poses and needs an IK solver behind it.

    Axis bindings:
        ============================== ==========================
        Joint 1 (shoulder_pan)         Left stick  left/right
        Joint 2 (shoulder_lift)        Left stick  up/down
        Joint 3 (elbow_flex)           Right stick up/down
        Joint 4 (wrist_flex)           Right stick left/right
        Joint 5 (wrist_roll)           D-pad       left/right
        Joint 6 (gripper)              Triggers    RT open / LT close
        ============================== ==========================

    Buttons mirror the keyboard's session controls:
        A       start control   (keyboard B)
        X       start recording (keyboard S)
        B       discard episode (keyboard D)
        Y       success + save  (keyboard N)

    Sticks are analogue, so the delta scales with deflection — unlike the keyboard,
    where a key is fully on or fully off. Small corrections near the ring are the whole
    reason for preferring a pad here.
    """

    def __init__(self, env, sensitivity: float = 0.05, dead_zone: float = 0.12):
        super().__init__(env)
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        # weakref so this object stays collectable despite the subscription
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )
        self._create_key_bindings()

        # Each axis reports as two one-sided inputs (e.g. LEFT_STICK_UP and
        # LEFT_STICK_DOWN), and releasing one sends a 0 for that direction only. Holding
        # both halves per joint and subtracting is what keeps a release from being lost.
        self._axis_raw = np.zeros((2, 6))  # [positive, negative] x 6 joints

        self.started = False
        self._reset_state = 0
        self._additional_callbacks = {}

    def __del__(self):
        if getattr(self, "_gamepad_sub", None) is not None:
            self._input.unsubscribe_to_gamepad_events(self._gamepad, self._gamepad_sub)
            self._gamepad_sub = None

    def __str__(self) -> str:
        name = self._input.get_gamepad_name(self._gamepad) if self._gamepad else "none"
        return (
            "Gamepad Controller for SO-101 joints.\n"
            f"\tDevice name: {name}\n"
            "\t----------------------------------------------\n"
            "\tJoint 1 (shoulder_pan):  Left stick  L/R\n"
            "\tJoint 2 (shoulder_lift): Left stick  U/D\n"
            "\tJoint 3 (elbow_flex):    Right stick U/D\n"
            "\tJoint 4 (wrist_flex):    Right stick L/R\n"
            "\tJoint 5 (wrist_roll):    D-pad       L/R\n"
            "\tJoint 6 (gripper):       RT open / LT close\n"
            "\t----------------------------------------------\n"
            "\tStart Control: A\n"
            "\tStart Recording: X\n"
            "\tDiscard Episode: B\n"
            "\tTask Success and Save: Y\n"
            "\tControl+C: quit"
        )

    # ── Device interface ─────────────────────────────────────────────────────────

    def get_device_state(self):
        return self._axis_raw[0] - self._axis_raw[1]

    def input2action(self):
        reset = self._reset_state
        ac_dict = {"reset": reset, "started": self.started, "keyboard": True}
        if reset:
            self._reset_state = False
            return ac_dict
        # Same key the keyboard branch of preprocess_device_action reads: both devices
        # hand over a 6-vector of joint deltas, so nothing downstream needs to change.
        ac_dict["joint_state"] = self.get_device_state() * self.sensitivity
        return ac_dict

    def reset(self):
        self._axis_raw = np.zeros((2, 6))

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    # ── carb plumbing ────────────────────────────────────────────────────────────

    def _on_gamepad_event(self, event, *args, **kwargs):
        value = event.value
        if abs(value) < self.dead_zone:
            value = 0.0

        if event.input in self._AXIS_BINDINGS:
            sign, joint = self._AXIS_BINDINGS[event.input]
            self._axis_raw[sign, joint] = value
            return True

        # Buttons report 1.0 on press and 0.0 on release; act on the press only.
        if event.input in self._BUTTON_BINDINGS and event.value > 0.5:
            name = self._BUTTON_BINDINGS[event.input]
            if name == "START":
                self.started = True
                self._reset_state = False
            elif name in self._additional_callbacks:
                self._additional_callbacks[name]()
        return True

    def _create_key_bindings(self):
        G = carb.input.GamepadInput
        # (index into [positive, negative], joint index)
        #
        # shoulder_lift and elbow_flex are INVERTED on purpose. Measured on this asset by
        # holding a steady delta and watching the jaw's world z:
        #     shoulder_lift +0.05 -> z 0.617 to 0.608 (down),  -0.05 -> 0.622 to 0.631 (up)
        #     elbow_flex    +0.05 -> z 0.622 to 0.614 (down),  -0.05 -> 0.619 to 0.827 (up)
        # so a positive command on either lowers the gripper. Since the action is
        # `current_position + delta`, binding stick-up to positive made the arm sink for
        # as long as the stick was held. Stick up now means gripper up.
        self._AXIS_BINDINGS = {
            G.LEFT_STICK_RIGHT: (0, 0), G.LEFT_STICK_LEFT: (1, 0),   # shoulder_pan
            G.LEFT_STICK_UP: (1, 1), G.LEFT_STICK_DOWN: (0, 1),      # shoulder_lift (inverted)
            G.RIGHT_STICK_UP: (1, 2), G.RIGHT_STICK_DOWN: (0, 2),    # elbow_flex (inverted)
            G.RIGHT_STICK_RIGHT: (0, 3), G.RIGHT_STICK_LEFT: (1, 3),  # wrist_flex
            G.DPAD_RIGHT: (0, 4), G.DPAD_LEFT: (1, 4),               # wrist_roll
            G.RIGHT_TRIGGER: (0, 5), G.LEFT_TRIGGER: (1, 5),         # gripper
        }
        # Names match the keyboard's callback keys registered by register_teleop_callbacks
        self._BUTTON_BINDINGS = {
            G.A: "START",
            G.X: "S",
            G.B: "D",
            G.Y: "N",
        }
