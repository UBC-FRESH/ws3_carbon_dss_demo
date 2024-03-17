# ws3_libcbm_dss

This repository contains all the files necessary to deploy a proof-of-concept example of a decision-support-system (DSS) prototype combining ws3 and libcbm_py (with a Jupyter notebook interface).

# Deployment and Installation

Note that the instructions here assume that you are deploying this package in a Linux environment. Development and testing was in a Ubuntu 22.04 environment.

Although not strictly required, we recommend installing this package into a Python venv (virtual environment) to minimize interactions with system-level packages. We provide instructions for how to do this below.

If you have not already done so, clone this repository to your local host (we clone to the home directory in the example below, but can be deployed anywhere the user has full read-write-execute permissions).

```bash
git clone https://github.com/UBC-FRESH/eccc_nscsf_dss.git
```

Create a new local Python virtual environment named `.venv`, and activate the new virtual environment.

```bash
mkdir .venv
python -m venv .venv/foo
source .venv/foo/bin/activate
```

Now install the required Python packages into the new virtual environment.

```back
python -m pip install -r eccc_nscsf_dss/requirements.txt
```

Make the new Python virtual environment available as a Python kernel from Jupyter notebooks.

```bash
python -m ipykernel install --user --name=venv
```

Log into your JupyterHub account, stop your Jupyter server, restart your Jupyter server. At this point you should be able to associate any of the Jupyter notebooks in this project with the new "venv" Python kernel, which should be configured correctly to enable all the code in the notebooks to run. 
