# ws3_libcbm_dss

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UBC-FRESH/ws3_carbon_dss_demo/main)

This repository contains all the files necessary to deploy a proof-of-concept example of a decision-support-system (DSS) prototype combining ws3 and libcbm_py (with a Jupyter notebook interface).

# Deployment and Installation

Note that the instructions here assume that you are deploying this package in a Linux environment. Development and testing was in a Ubuntu 22.04 environment.

Although not strictly required, we recommend installing this package into a Python venv (virtual environment) to minimize interactions with system-level packages. We provide instructions for how to do this below.

If you have not already done so, clone this repository to a working directory on your development machine. 
Required Python packages are listed in the 'requirements.txt' file in the root directory of the repository.

This repository includes a pre-configured Python virtual environment (`.venv`, in the root directory of the repository) that has all the Python packages required to run the Jupyter notebooks included in this project.
To use the included virtual environment, activate it and add a new Python kernel to your Jupyter computing envirnment linking to this virtual environment.

```bash
source .venv/bin/activate
python -m ipykernel install --user --name=venv
```

After running the above commands, log into your JupyterHub account, stop your Jupyter server, and restart your Jupyter server. 
At this point you should be able to run any of the Jupyter notebooks in this project with the new "venv" Python kernel, which should be configured correctly for the code in the notebooks to run. 

If you are using GitHub Codespaces, the `requirements.txt` file should be picked up automatically and the Codespaces environment _should_ bootstrap itself into a working state without needing any specific user input or configuration.
