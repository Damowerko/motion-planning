from gym.envs.registration import register

register(
    id='unlabeled-motion-planning-v0',
    entry_point='reconstrain.envs:UnlabeledMotionPlanning',
)