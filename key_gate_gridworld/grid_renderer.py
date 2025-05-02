
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

class GridWorldRenderer:
    def __init__(self, grid_size=4, asset_dir="assets"):
        self.grid_size = grid_size
        self.robot_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "robot.png")))
        self.key_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "key.png")))
        self.gate_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "gate.png")))
        self.goal_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "goal.png")))

    def render(self, agent_pos, has_key, key_pos, gate_pos, goal_pos, title="KeyGateGridWorld"):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(0, self.grid_size + 1))
        ax.set_yticks(np.arange(0, self.grid_size + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title)

        if not has_key and agent_pos != key_pos:
            self._draw_image(ax, self.key_img, key_pos)
        self._draw_image(ax, self.gate_img, gate_pos)
        self._draw_image(ax, self.goal_img, goal_pos)
        self._draw_image(ax, self.robot_img, agent_pos)

        plt.tight_layout()
        plt.show()

    def _draw_image(self, ax, img, position):
        x, y = map(int, position)
        ax.imshow(img, extent=(x, x + 1, y, y + 1), origin='upper', aspect='auto', zorder=10)

