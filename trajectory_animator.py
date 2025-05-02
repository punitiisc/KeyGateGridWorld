
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy as np
import os

class TrajectoryAnimator:
    def __init__(self, grid_size=4, asset_dir="assets"):
        self.grid_size = grid_size
        self.robot_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "robot.png")))
        self.key_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "key.png")))
        self.gate_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "gate.png")))
        self.goal_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "goal.png")))

    def animate(self, positions, rewards, key_pos, gate_pos, goal_pos, save_path=None):
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

        key_img = ax.imshow(self.key_img, extent=(key_pos[0], key_pos[0]+1, key_pos[1], key_pos[1]+1), origin='upper', zorder=2)
        gate_img = ax.imshow(self.gate_img, extent=(gate_pos[0], gate_pos[0]+1, gate_pos[1], gate_pos[1]+1), origin='upper', zorder=2)
        goal_img = ax.imshow(self.goal_img, extent=(goal_pos[0], goal_pos[0]+1, goal_pos[1], goal_pos[1]+1), origin='upper', zorder=2)
        robot_img = ax.imshow(self.robot_img, extent=(0, 1, 0, 1), origin='upper', zorder=3)

        reward_text = ax.text(self.grid_size + 0.2, 1, "", fontsize=12)

        def update(frame):
            x, y = positions[frame]
            robot_img.set_extent((x, x+1, y, y+1))
            reward_text.set_text(f"Reward: {rewards[frame]}")
            if (x, y) == key_pos and frame > 0:
                key_img.set_alpha(0.0)
            return robot_img, reward_text

        ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=700, blit=False, repeat=False)

        if save_path:
            ani.save(save_path, writer='pillow')
            print(f"Animation saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
