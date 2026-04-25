"""dimos — A fork of dimensionalOS/dimos.

A framework for building and orchestrating AI-powered robotic systems,
providing abstractions for agents, memory, perception, and actuation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dimos")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
