"""Local Qwen3.5-9B chooses an observed action. Code owns execution."""

from .agent import Agent
from .browser import Browser

__all__ = ["Agent", "Browser"]
