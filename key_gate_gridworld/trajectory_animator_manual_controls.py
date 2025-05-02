import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
from matplotlib.backends.backend_qt5 import TimerQT
import numpy as np
import os

class TrajectoryAnimator:
    def __init__(self, grid_size=4, asset_dir="assets"):
        self.grid_size = grid_size
        self.robot_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "robot.png")))
        self.key_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "key.png")))
        self.gate_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "gate.png")))
        self.goal_img = np.flipud(mpimg.imread(os.path.join(asset_dir, "goal.png")))

        self.frame = 0
        self.timer = None
        self.running = False

    def animate(self, positions, rewards, key_pos, gate_pos, goal_pos, save_path=None):
        self.positions = positions
        self.rewards = rewards
        self.key_pos = key_pos
        self.gate_pos = gate_pos
        self.goal_pos = goal_pos
        self.cum_rewards = np.cumsum(rewards)

        self.fig = plt.figure(figsize=(8, 11))
        gs = self.fig.add_gridspec(6, 1)
        self.ax_grid = self.fig.add_subplot(gs[:3])
        self.ax_plot = self.fig.add_subplot(gs[3:5])
        ax_controls = self.fig.add_subplot(gs[5])
        ax_controls.axis("off")
            
        self._setup_grid()
        self._setup_plot()
        self._setup_buttons()

        self.timer = self.fig.canvas.new_timer(interval=700)
        self.timer.add_callback(self._next_frame)

     
        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def _setup_grid(self):
        self.ax_grid.set_xlim(0, self.grid_size)
        self.ax_grid.set_ylim(0, self.grid_size)
        self.ax_grid.set_xticks(np.arange(0, self.grid_size + 1))
        self.ax_grid.set_yticks(np.arange(0, self.grid_size + 1))
        self.ax_grid.set_xticklabels([])
        self.ax_grid.set_yticklabels([])
        self.ax_grid.grid(True)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_title("KeyGateGridWorld")

        self.key_img_artist = self.ax_grid.imshow(self.key_img, extent=(self.key_pos[0], self.key_pos[0]+1, self.key_pos[1], self.key_pos[1]+1), origin='upper', zorder=2)
        self.gate_img_artist = self.ax_grid.imshow(self.gate_img, extent=(self.gate_pos[0], self.gate_pos[0]+1, self.gate_pos[1], self.gate_pos[1]+1), origin='upper', zorder=2)
        self.goal_img_artist = self.ax_grid.imshow(self.goal_img, extent=(self.goal_pos[0], self.goal_pos[0]+1, self.goal_pos[1], self.goal_pos[1]+1), origin='upper', zorder=2)
        self.robot_img_artist = self.ax_grid.imshow(self.robot_img, extent=(0, 1, 0, 1), origin='upper', zorder=3)

    def _setup_plot(self):
        self.reward_line, = self.ax_plot.plot([], [], color='blue')
        self.reward_text = self.ax_plot.text(0.01, 0.95, "", transform=self.ax_plot.transAxes, ha='left', va='top', fontsize=10)
        self.step_text = self.ax_plot.text(0.99, 0.95, "", transform=self.ax_plot.transAxes, ha='right', va='top', fontsize=10)
        self.ax_plot.set_xlim(0, len(self.rewards))
        self.ax_plot.set_ylim(min(0, self.cum_rewards.min()), self.cum_rewards.max() + 5)
        self.ax_plot.set_title("Cumulative Reward")
        self.ax_plot.set_xlabel("Step")
        self.ax_plot.set_ylabel("Total Reward")

    def _setup_buttons(self):
        ax_play = plt.axes([0.2, 0.01, 0.1, 0.04])
        ax_pause = plt.axes([0.31, 0.01, 0.1, 0.04])
        ax_rewind = plt.axes([0.42, 0.01, 0.1, 0.04])
        ax_forward = plt.axes([0.53, 0.01, 0.1, 0.04])

        Button(ax_play, "Play").on_clicked(self._start_timer)
        Button(ax_pause, "Pause").on_clicked(self._pause_timer)
        Button(ax_rewind, "← Back").on_clicked(self._step_back)
        Button(ax_forward, "Next →").on_clicked(self._step_forward)

    def _next_frame(self):
        if self.running and self.frame < len(self.positions) - 1:
            self.frame += 1
            self._update_plot()

    def _update_plot(self):
        x, y = self.positions[self.frame]
        self.robot_img_artist.set_extent((x, x + 1, y, y + 1))
        self.reward_line.set_data(range(self.frame + 1), self.cum_rewards[:self.frame + 1])
        self.reward_text.set_text(f"Step Reward: {self.rewards[self.frame]}")
        self.step_text.set_text(f"Cumulative: {self.cum_rewards[self.frame]}")
        if (x, y) == self.key_pos and self.frame > 0:
            self.key_img_artist.set_alpha(0.0)
        self.fig.canvas.draw_idle()

    def _start_timer(self, event):
        if not self.running:
            self.running = True
            self.timer.start()

    def _pause_timer(self, event):
        self.running = False
        self.timer.stop()

    def _step_back(self, event):
        if self.frame > 0:
            self.frame -= 1
            self._update_plot()

    def _step_forward(self, event):
        if self.frame < len(self.positions) - 1:
            self.frame += 1
            self._update_plot()
