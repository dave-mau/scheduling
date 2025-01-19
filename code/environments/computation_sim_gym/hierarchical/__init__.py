import gymnasium as gym

from .builder import HierarchicalSystemBuilder
from .hierarchical_system_v0 import HierarchicalSystem, InformationLossObserver, ParsedHierarchicalSystem
from .reward import Reward
from .types import ActionCollection, SystemCollection
from .config_parser import ConfigParser

gym.register(
    "HierarchicalSystem-v0",
    entry_point=HierarchicalSystem,
)
gym.register(
    "ParsedHierarchicalSystem-v0",
    entry_point=ParsedHierarchicalSystem,
)
