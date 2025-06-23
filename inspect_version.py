#!/usr/bin/env python3
import sys, subprocess, json, os
from pathlib import Path

def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.stdout.strip(), p.stderr.strip()

info = {}

# 1) Python version & executable
info['python_executable'] = sys.executable
info['python_version']    = sys.version.replace("\n"," ")

# 2) Virtualenv root (if any)
info['venv_root'] = os.environ.get('VIRTUAL_ENV', '— not in a venv —')

# 3) Installed packages (sinergym, EnergyPlus API, etc.)
try:
    import pkg_resources
    pkgs = {d.project_name: d.version for d in pkg_resources.working_set}
    info['installed_packages'] = {k: pkgs[k] for k in sorted(pkgs)
                                  if k.lower() in ('sinergym','energyplus','numpy','torch','gym')}
except ImportError:
    info['installed_packages'] = 'pkg_resources unavailable'

# 4) Where Sinergym lives on disk
try:
    import sinergym
    path = Path(sinergym.__file__).resolve()
    info['sinergym_path'] = str(path.parent)
except ImportError:
    info['sinergym_path'] = 'not installed'

# 5) EnergyPlus version & location
ep_out, ep_err = run("which energyplus")
info['energyplus_binary'] = ep_out or ep_err
ep_out, ep_err = run("energyplus --version")
info['energyplus_version'] = ep_out or ep_err

# 6) Summarize
print(json.dumps(info, indent=2))