"""Bridge to DEAL's native sparse-GP extension.

The Python active-learning surface lives under ``deal.sgp``.  Prefer DEAL's
own native module, and keep the FLARE import only as a transitional fallback
for environments that have not rebuilt DEAL yet.
"""

try:
    from ._C_deal_sgp import *  # noqa: F401,F403
except ImportError:
    from flare.bffs.sgp._C_flare import *  # noqa: F401,F403
