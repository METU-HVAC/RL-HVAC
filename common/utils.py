
import os
import shutil
#Before starting, clear any directories that may have been created in previous runs
#They start with Eplus-env-* and are created by EnergyPlus
def remove_previous_run_logs():
    for root, dirs, files in os.walk(".", topdown=False):
        for name in dirs:
            if name.startswith("Eplus-env-"):
                dir_path = os.path.join(root, name)
                shutil.rmtree(dir_path)
