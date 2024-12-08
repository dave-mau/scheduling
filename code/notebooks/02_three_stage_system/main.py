import sys
import gymnasium as gym

sys.path.insert(0, "/home/davidmauderli/repos/scheduling/code/")
from system_config import SystemConfig


env = gym.make("HierarchicalSystem-v0", **SystemConfig().make(), render_mode="human").unwrapped
env.reset(0)
for i in range(1000):
    env.step(3)
    input("Press Enter to continue...")
