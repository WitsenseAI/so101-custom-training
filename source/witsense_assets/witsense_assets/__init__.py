import os
import toml
from pathlib import Path

try:
    from git import Repo
    repo= Repo(os.getcwd(), search_parent_directories =True)
    git_root= repo.git.rev_parse("--show-toplevel")
except Exception:
    git_root= Path(os.path.abspath(__file__)).parent.parent.parent.parent.parent

MAIN_ASSETS_DIR= os.path.join(git_root, "Assets")
'''this i sthe path to the assets directory that we download using huggingface cli'''

WITSENSE_ASSETS_EXT_DIR= Path(os.path.abspath(__file__)).parent
'''this is the path to the source directory'''



