
# 🔑 KeyGateGridWorld 🟩

A custom [Gymnasium](https://gymnasium.farama.org/) 4x4 GridWorld environment where an agent must:
- Collect a key 🔑
- Pass through a locked gate 🚪
- Reach a goal 🏁

Includes real-time **trajectory visualizations** with cumulative reward plots.

---

## 🚀 Features

- 🔁 **Deterministic or stochastic dynamics**
- 🧭 Reward shaping: step penalty, key bonus, goal reward
- 📊 Matplotlib-based visualization with trajectory and reward graph
- 🧪 Designed for experimentation and teaching reinforcement learning basics

---

## 📦 Installation

Clone the repo and install in editable mode:

```bash
git clone git@github.com:punitguptaiisc/KeyGateGridWorld.git
cd KeyGateGridWorld
pip install -e .
```

---

## 🧪 Run Sample Trajectories

```bash
python trajectory_controls_with_save.py
```

You’ll see animated trajectories of agents in the grid collecting keys and rewards.
Please note that in the first run dummy screens appear(for confirmation) but however trajectories are created and also saved for playing later

---

## 🧰 Project Structure

```
KeyGateGridWorld/
├── key_gate_gridworld/             # Custom Gymnasium environment
│   ├── key_gate_gridworld_env.py
│   ├── trajectory_animator_with_plot.py
│   └── assets/                     # Icons for key, gate, goal, robot
├── trajectory_controls_with_save.py
├── README.md
├── setup.py
└── requirements.txt
```

---

## 📸 Preview

You can capture animated `.gif` files for each trajectory or embed visualizations in notebooks.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
