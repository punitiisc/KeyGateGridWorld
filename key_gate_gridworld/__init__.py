from gymnasium.envs.registration import register

register(
    id="KeyGateGridWorld-v0",
    entry_point="key_gate_gridworld.key_gate_gridworld_env:KeyGateGridWorld",
)
