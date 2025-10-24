from gymnasium.core import Wrapper


class ActionRepeat(Wrapper):
    def __init__(self, env, repeat):
        assert repeat >= 1, "Expects at least one repeat."
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0.0
        total_cost = 0.0
        current_step = 0
        info = {"steps": 0}
        while current_step < self.repeat and not done:
            obs, reward, terminal, truncated, info = self.env.step(action)
            total_reward += reward
            total_cost += info.get("cost", 0.0)
            current_step += 1
            done = truncated or terminal
        info["steps"] = current_step
        info["cost"] = total_cost
        return obs, total_reward, terminal, truncated, info

class Autoreset(Wrapper):
    def __init__(self, env):
        super(Autoreset, self).__init__(env)
        self.autoreset = False

    def reset(self, *, seed = None, options = None):
        self.autoreset = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.autoreset:
            obs, info = self.env.reset()
            reward, terminated, truncated = 0.0, False, False
            info["cost"] = info.get("cost", 0.0)
            info["steps"] = 0
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)

        self.autoreset = terminated or truncated
        return obs, reward, terminated, truncated, info