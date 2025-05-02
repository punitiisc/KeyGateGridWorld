import gymnasium as gym
import key_gate_gridworld
from key_gate_gridworld.grid_renderer import GridWorldRenderer

env = gym.make("KeyGateGridWorld-v0", stochastic_env=False)
obs, _ = env.reset()

positions = [tuple(obs["agent"])]
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    positions.append(tuple(obs["agent"]))
    if done or truncated:
        break

env.close()

# Visualize last state using icons
start_pos = (0, 0)
obs, _ = env.reset()

renderer = GridWorldRenderer(grid_size=4, asset_dir="key_gate_gridworld/assets")
renderer.render(agent_pos=start_pos, has_key=obs["has_key"],
                key_pos=(1, 2), gate_pos=(2, 2), goal_pos=(3, 3))

