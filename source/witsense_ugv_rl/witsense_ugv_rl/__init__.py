import os
import toml

# Conveniences to other module directories via relative paths
WITSENSE_UGV_RL_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
"""Path to the extension source directory."""

WITSENSE_UGV_RL_LOGS_DIR = os.path.join(WITSENSE_UGV_RL_EXT_DIR, "logs")
"""Path to the extension data directory."""