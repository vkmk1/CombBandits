from .base import Agent
from .cucb import CUCBAgent
from .cts import CTSAgent
from .llm_cucb_at import LLMCUCBATAgent
from .llm_greedy import LLMGreedyAgent
from .ellm_adapted import ELLMAdaptedAgent
from .opro_bandit import OPROBanditAgent
from .corrupt_robust_cucb import CorruptRobustCUCBAgent
from .warm_start_cts import WarmStartCTSAgent
from .exp4 import EXP4Agent

AGENT_REGISTRY = {
    "cucb": CUCBAgent,
    "cts": CTSAgent,
    "llm_cucb_at": LLMCUCBATAgent,
    "llm_greedy": LLMGreedyAgent,
    "ellm_adapted": ELLMAdaptedAgent,
    "opro_bandit": OPROBanditAgent,
    "corrupt_robust_cucb": CorruptRobustCUCBAgent,
    "warm_start_cts": WarmStartCTSAgent,
    "exp4": EXP4Agent,
}

__all__ = ["Agent", "AGENT_REGISTRY"] + list(AGENT_REGISTRY.keys())
