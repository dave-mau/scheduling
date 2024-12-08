import gymnasium as gym

from .builder import HierarchicalSystemBuilder
from .hierarchical_system_v0 import HierarchicalSystem, InformationLossObserver
from .reward import Reward
from .types import ActionCollection, SystemCollection

gym.register(
    "HierarchicalSystem-v0",
    entry_point=HierarchicalSystem,
)
