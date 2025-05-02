
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np
import os

class TrajectoryAnimator:
    def __init__(self, grid_size=4, asset_dir="assets"):
        self.grid_size = grid_size
        self.robot_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "robot.png")))
        self.key_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "key.png")))
        self.gate_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "gate.png")))
        self.goal_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "goal.png")))
        self.ani = None
        self.paused = False
        self.frame = 0

    def animate(self, positions, rewards, key_pos, gate_pos, goal_pos, save_path=None):
        fig = plt.figure(figsize=(8, 11))
        gs = fig.add_gridspec(6, 1)
        ax_grid = fig.add_subplot(gs[:3])
        ax_plot = fig.add_subplot(gs[3:5])
        ax_controls = fig.add_subplot(gs[5])
        ax_controls.axis("off")

        ax_grid.set_xlim(0, self.grid_size)
        ax_grid.set_ylim(0, self.grid_size)
        ax_grid.set_xticks(np.arange(0, self.grid_size + 1))
        ax_grid.set_yticks(np.arange(0, self.grid_size + 1))
        ax_grid.set_xticklabels([])
        ax_grid.set_yticklabels([])
        ax_grid.grid(True)
        ax_grid.set_aspect('equal')
        ax_grid.invert_yaxis()
        ax_grid.set_title("KeyGateGridWorld")

        key_img = ax_grid.imshow(self.key_img, extent=(key_pos[0], key_pos[0]+1, key_pos[1], key_pos[1]+1), origin='upper', zorder=2)
        gate_img = ax_grid.imshow(self.gate_img, extent=(gate_pos[0], gate_pos[0]+1, gate_pos[1], gate_pos[1]+1), origin='upper', zorder=2)
        goal_img = ax_grid.imshow(self.goal_img, extent=(goal_pos[0], goal_pos[0]+1, goal_pos[1], goal_pos[1]+1), origin='upper', zorder=2)
        robot_img = ax_grid.imshow(self.robot_img, extent=(0, 1, 0, 1), origin='upper', zorder=3)

        cum_rewards = np.cumsum(rewards)
        reward_line, = ax_plot.plot([], [], color='blue')
        reward_text = ax_plot.text(0.01, 0.95, "", transform=ax_plot.transAxes, ha='left', va='top', fontsize=10)
        step_text = ax_plot.text(0.99, 0.95, "", transform=ax_plot.transAxes, ha='right', va='top', fontsize=10)

        ax_plot.set_xlim(0, len(rewards))
        ax_plot.set_ylim(min(0, cum_rewards.min()), cum_rewards.max() + 5)
        ax_plot.set_title("Cumulative Reward")
        ax_plot.set_xlabel("Step")
        ax_plot.set_ylabel("Total Reward")

        def update_plot():
            x, y = positions[self.frame]
            robot_img.set_extent((x, x+1, y, y+1))
            reward_line.set_data(range(self.frame + 1), cum_rewards[:self.frame + 1])
            reward_text.set_text(f"Step Reward: {rewards[self.frame]}")
            step_text.set_text(f"Cumulative: {cum_rewards[self.frame]}")
            if (x, y) == key_pos and self.frame > 0:
                key_img.set_alpha(0.0)
            fig.canvas.draw_idle()

        def play(event): self.paused = False
        def pause(event): self.paused = True
        def rewind(event):
            self.frame = max(0, self.frame - 1)
            update_plot()
        def forward(event):
            self.frame = min(len(positions) - 1, self.frame + 1)
            update_plot()

        ax_play = plt.axes([0.25, 0.01, 0.1, 0.04])
        ax_pause = plt.axes([0.36, 0.01, 0.1, 0.04])
        ax_rewind = plt.axes([0.47, 0.01, 0.1, 0.04])
        ax_forward = plt.axes([0.58, 0.01, 0.1, 0.04])

        Button(ax_play, "Play").on_clicked(play)
        Button(ax_pause, "Pause").on_clicked(pause)
        Button(ax_rewind, "← Back").on_clicked(rewind)
        Button(ax_forward, "Next →").on_clicked(forward)

        def update(_):
            if not self.paused and self.frame < len(positions) - 1:
                self.frame += 1
                update_plot()
            return []

        self.ani = animation.FuncAnimation(fig, update, interval=700, blit=False, repeat=False)

        if save_path:
            self.ani.save(save_path, writer='pillow')
            print(f"Animation saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
