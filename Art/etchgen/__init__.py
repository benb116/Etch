"""etchgen: image -> single continuous line ("lineset") generation.

See PLAN.md for the architecture. Core idea: all art lives in the vector
domain as lists of polylines; a topological graph solver joins and
Eulerizes them into one unbroken stroke with minimal visible extra ink.
"""

__version__ = "0.1.0"
