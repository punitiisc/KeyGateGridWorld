# KeyGateGridWorld Project

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class KeyGateGridWorld(gym.Env):
    """
    A 4x4 GridWorld with a Key, Locked Gate, and Goal.
    The agent must collect the key before passing the locked gate to reach the goal.
    Supports deterministic and stochastic environment settings.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, stochastic_env=True, discount_factor=0.99):
        super().__init__()

        self.grid_size = 4
        self.start_pos = (0, 0)
        self.key_pos = (1, 2)
        self.gate_pos = (2, 2)
        self.goal_pos = (3, 3)
        self.discount = discount_factor

        self.stochastic_env = stochastic_env
        self.slip_prob = 0.1 if stochastic_env else 0.0

        self.observation_space = spaces.Dict({
            "agent": spaces.MultiDiscrete([self.grid_size, self.grid_size]),
            "has_key": spaces.Discrete(2)
        })

        self.action_space = spaces.Discrete(4)  # 0=Up, 1=Down, 2=Left, 3=Right

        self.agent_pos = self.start_pos
        self.has_key = False
        self.steps = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.has_key = False
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return {"agent": np.array(self.agent_pos, dtype=np.int32), "has_key": int(self.has_key)}

    def step(self, action):
        self.steps += 1

        if np.random.rand() < self.slip_prob:
            action = self.action_space.sample()

        x, y = self.agent_pos
        new_x, new_y = x, y

        if action == 0 and y > 0:
            new_y -= 1
        elif action == 1 and y < self.grid_size - 1:
            new_y += 1
        elif action == 2 and x > 0:
            new_x -= 1
        elif action == 3 and x < self.grid_size - 1:
            new_x += 1

        attempted_pos = (new_x, new_y)
        reward = -1  # step penalty

        if attempted_pos == self.gate_pos and not self.has_key:
            attempted_pos = self.agent_pos
            reward -= 1  # penalty for blocked gate

        if attempted_pos == self.agent_pos:
            reward -= 1  # penalty for no movement

        self.agent_pos = attempted_pos

        if self.agent_pos == self.key_pos and not self.has_key:
            self.has_key = True
            reward += 2  # reward for key

        terminated = self.agent_pos == self.goal_pos
        truncated = self.steps >= self.max_steps

        if terminated:
            reward = 10  # reward for reaching goal

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        x, y = self.agent_pos
        grid[y][x] = "A"
        kx, ky = self.key_pos
        gx, gy = self.goal_pos
        lx, ly = self.gate_pos
        if not self.has_key:
            grid[ky][kx] = "K"
        grid[ly][lx] = "L"
        grid[gy][gx] = "G"
        print("\n".join([" ".join(row) for row in grid]) + "\n")

    def close(self):
        pass
