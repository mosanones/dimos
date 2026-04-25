"""dimos — A fork of dimensionalOS/dimos.

A framework for building and orchestrating AI-powered robotic systems,
providing abstractions for agents, memory, perception, and actuation.

Personal fork notes:
    - Forked for personal learning and experimentation.
    - Upstream: https://github.com/dimensionalOS/dimos
    - Using a fallback version of "unknown" instead of "0.0.0.dev0" so it's
      clearer when the package isn't properly installed vs. a real dev build.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dimos")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["__version__"]
