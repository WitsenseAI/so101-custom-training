from .rss_cfgs import *

from witsense_ugv_rl.utils.hydra import register_run_to_hydra

# Only the elevation task is registered in witsense_ugv_tasks. The drift and visual run
# configs exist upstream but name gym ids this repo does not register.
register_run_to_hydra("RSS_ELEV_CONFIG", RSS_ELEV_CONFIG)
