
import gymnasium as gym
import key_gate_gridworld

from key_gate_gridworld.trajectory_animator_manual_controls import TrajectoryAnimator


def run_single_episode(env):
    obs, _ = env.reset()
    positions = [tuple(obs["agent"])]
    rewards = [0]
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        positions.append(tuple(obs["agent"]))
        rewards.append(reward)
        if done or truncated:
            break
    return positions, rewards

def main():
    env = gym.make("KeyGateGridWorld-v0", stochastic_env=True)
    animator = TrajectoryAnimator(grid_size=4, asset_dir="key_gate_gridworld/assets")

    for episode in range(10):
        print(f"▶️ Generating and Saving Trajectory {episode + 1}")
        positions, rewards = run_single_episode(env)
        animator.animate(
            positions=positions,
            rewards=rewards,
            key_pos=(1, 2),
            gate_pos=(2, 2),
            goal_pos=(3, 3),
            save_path=f"trajectory_{episode+1}.gif"
        )

    env.close()

if __name__ == "__main__":
    main()
