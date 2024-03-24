from gym.envs.registration import register

register(
    id="motion-planning-v0",
    entry_point="motion_planning.envs.motion_planning:MotionPlanning",
)
