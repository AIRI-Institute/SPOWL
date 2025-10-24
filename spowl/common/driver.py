import numpy as np
from tqdm import tqdm

from spowl.common.episodic_async_env import EpisodicAsync
from spowl.common.trajectory import Trajectory, TrajectoryData
from spowl.common.types import Agent, Report


def _summarize_episodes(trajectory: TrajectoryData, steps: int) -> Report:
    done = np.logical_or(trajectory.terminated, trajectory.truncated)
    num_episodes = done[:, :steps].sum()
    reward = float(trajectory.reward[:, :steps].sum() / num_episodes)
    total_cost = float(trajectory.cost[:, :steps].sum())
    cost = total_cost/ num_episodes
    return Report({'reward_return': reward, 'cost_return': cost, 'total_cost': total_cost})


def _summarize_eval_episodes(trajectory: Trajectory) -> Report:
    data = trajectory.transitions
    done = np.logical_or(data.terminated, data.truncated)
    flag = np.ones(done.shape[0])
    reward = np.zeros(done.shape[0])
    cost = np.zeros(done.shape[0])
    length = np.zeros(done.shape[0])
    for i in range(done.shape[1]):
        reward += data.reward[:, i, 0] * flag
        cost += data.cost[:, i, 0] * flag
        length += flag
        flag = np.logical_and(flag, np.logical_not(done[:, i, 0]))
    
    video_dict = {}
    if len(trajectory.frames) > 0:
        video = np.asarray(trajectory.frames)
        video = np.transpose(video, (0, 2, 1, 3, 4))
        video = np.reshape(video, (*video.shape[:2], video.shape[2] * video.shape[3], video.shape[-1]))
        video = np.transpose(video, (0, 3, 1, 2))
        video_dict['video'] = video
    return Report(
        {'reward_return': reward, 'cost_return': cost, 'lengths': length - 1},
        video_dict
    )


def interact(
    agent: Agent,
    environment: EpisodicAsync,
    episodes: int,
    train: bool,
    step: int,
    render_episodes: int = 0,
) -> tuple[Trajectory, int, list]:
    
    parallel_steps = episodes * (environment.time_limit // environment.action_repeat + 1)
    collect_steps = agent.config.training.collect_steps
    add_index = 0
    trajectory = Trajectory(TrajectoryData(
        np.zeros((environment.num_envs, parallel_steps, environment.observation_space.shape[-1]), dtype=np.float32),
        np.zeros((environment.num_envs, parallel_steps, environment.action_space.shape[-1]), dtype=np.float32),
        np.zeros((environment.num_envs, parallel_steps, 1), dtype=np.float32),
        np.zeros((environment.num_envs, parallel_steps, 1), dtype=np.float32),
        np.zeros((environment.num_envs, parallel_steps, 1), dtype=np.bool_),
        np.zeros((environment.num_envs, parallel_steps, 1), dtype=np.bool_),
    ))
    
    observations = environment.reset()
    step_count = 0
    is_first = np.ones(environment.num_envs)
    done = np.zeros(environment.num_envs)
    # Set initial observation
    trajectory.transitions.observation[:, 0] = observations
    with tqdm(total=environment.time_limit * episodes * environment.num_envs, unit=f" Step") as pbar:
        while step_count < (parallel_steps - 1):
            render = render_episodes > 0
            if render:
                trajectory.frames.append(environment.render())
            actions = np.array([environment.action_space.sample()] * environment.num_envs)
            actions = agent(observations, train, step, is_first)
            observations, rewards, terminated, truncated, infos = environment.step(actions)
            is_first[:] = done
            done = np.logical_or(terminated, truncated)
            costs = np.array([info.get("cost", 0) for info in infos])
            trajectory.add_transition(observations, actions, rewards, costs, terminated, truncated)
            steps_in_envs = sum([info['steps'] for info in infos])
            pbar.update(steps_in_envs)
            step_count += 1
            step += steps_in_envs
            
            # Add intermediate data
            if train and step_count % collect_steps == 0:
                assert add_index + collect_steps <= trajectory.index
                agent.add_data(trajectory.transitions, add_index, add_index + collect_steps)
                add_index += collect_steps
            if done.any():
                summary = _summarize_episodes(trajectory.transitions, trajectory.index)
                pbar.set_postfix({"reward": summary.metrics['reward_return'], "cost": summary.metrics['cost_return']})
                if render:
                    render_episodes = max(render_episodes - 1, 0)
                
    # Set final flag (important for variable episode lengths)
    trajectory.transitions.truncated[:, -1] = True
    if train:
        agent.add_data(trajectory.transitions, add_index, trajectory.index)
    return trajectory, step


def epoch(
    agent: Agent,
    env: EpisodicAsync,
    parallel_steps: int,
    train: bool,
    step: int,
    render_episodes: int = 0,
) -> tuple[Report, int]:
    data, step = interact(agent, env, parallel_steps, train, step, render_episodes if not train else 0)
    if train:
        summary = _summarize_episodes(data.transitions, data.index)
    else:
        summary = _summarize_eval_episodes(data)
    return summary, step