import json
import os
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import TimeLimit as TimeLimitWrapper
from computation_sim_gym import hierarchical
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.ppo import PPO


def make_env(max_timestep_episode, render_mode=None) -> gym.Env:
    config_path = Path(__file__).absolute().parent / "system_config.json"
    env = gym.make(
        "ParsedHierarchicalSystem-v0",
        system_config_file=config_path,
        render_mode=render_mode,
    ).unwrapped
    check_env(env)
    return TimeLimitWrapper(env, max_timestep_episode)


def make_envs(
    n_envs: int, seed: int, monitor_dir: Path, episode_length: int, render_mode=None, wrap_norm=True
) -> DummyVecEnv:
    monitor_kwargs = dict(
        allow_early_resets=True,
        info_keywords=(
            "buffer_overrides",
            "missing_inputs",
            "missing_measurements",
            "output_age_avg",
            "output_age_max",
            "output_age_min",
        ),
    )
    venv = make_vec_env(
        lambda: make_env(episode_length, render_mode=render_mode),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(monitor_dir),
        vec_env_cls=DummyVecEnv,
        monitor_kwargs=monitor_kwargs,
    )
    # venv = VecFrameStack(venv, n_stack = 10) # necessary?
    if wrap_norm:
        venv = VecNormalize(venv, training=True, norm_obs=True, norm_reward=True)
    return venv


class SaveVecNormalize(EventCallback):
    def on_step(self):
        vec_norm_env = self.parent.model.get_vec_normalize_env()
        if not vec_norm_env:
            raise ValueError("There is not vec normalize environment to save.")
        if self.parent.best_model_save_path is not None:
            vec_norm_env.save(os.path.join(self.parent.best_model_save_path, "best_vec_norm"))
        return True


class SavePaths:
    def __init__(self):
        self.cwd = Path(__file__).absolute().parent
        self.logs = self.cwd / "logs"
        self.train = self.logs / "train"
        self.tensorboard = self.train / "tensorboard"
        self.eval = self.logs / "eval"
        self.train_monitor = self.train / "monitor"
        self.eval_monitor = self.eval / "monitor"
        self.best_model = self.eval / "best_model"
        self.eval_logs = self.eval / "log"


def main(params, paths: SavePaths):
    envs_train = make_envs(params["n_envs_train"], params["seed_train"], paths.train_monitor, params["episode_length"])
    envs_eval = make_envs(params["n_envs_eval"], params["seed_eval"], paths.eval_monitor, params["episode_length"])
    eval_callback = EvalCallback(
        envs_eval,
        best_model_save_path=paths.best_model,
        log_path=paths.eval_logs,
        eval_freq=max(params["freq_eval"] // params["n_envs_train"], 1),
        n_eval_episodes=params["n_episodes_eval"],
        deterministic=True,
        render=False,
        verbose=1,
        callback_on_new_best=SaveVecNormalize(),
    )
    model = PPO("MlpPolicy", n_steps=512, env=envs_train, device="cpu", verbose=0, tensorboard_log=paths.tensorboard)
    model.learn(
        total_timesteps=params["total_timesteps_train"], callback=eval_callback, progress_bar=True, tb_log_name="ppo"
    )


def run(params, paths: SavePaths):
    env: DummyVecEnv = make_envs(
        1, 10, None, 100, render_mode="human", wrap_norm=False  # n-envs  # seed  # monitor dir  # ep length
    )
    env = VecNormalize.load(paths.best_model / "best_vec_norm", env)
    model = PPO.load(paths.best_model / "best_model.zip", env=env)
    states = None
    observation = env.reset()
    for i in range(1000):
        actions, states = model.predict(
            observation,  # type: ignore[arg-type]
            state=states,
            deterministic=True,
        )
        observation, rewards, dones, infos = env.step(actions)
        input("Press Enter to continue...")


if __name__ == "__main__":
    params = {
        "n_envs_train": 8,
        "n_envs_eval": 1,
        "n_episodes_eval": 30,
        "seed_train": 100,
        "seed_eval": 1000,
        "freq_eval": int(10 * 60 * 1_000 / 10),
        "total_timesteps_train": int(3 * 60 * 60 * 1_000 / 10),
        "episode_length": int(10 * 1_000 / 10),
    }
    paths = SavePaths()
    main(params, paths)
    run(params, paths)
