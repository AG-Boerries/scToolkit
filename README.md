# scToolkit

## Disclaimer

**scToolkit** is an independent project and is **not affiliated with, endorsed by, or part of the Scanpy project** or the scverse ecosystem.  
This toolkit provides a collection of convenience and helper functions that operate **on top of Scanpy**, but it does **not** replace or modify Scanpy itself.

To use scToolkit, you must have **Scanpy installed separately** (see the installation section for details).  
All references to “Scanpy” in this repository are solely for the purpose of describing compatibility and required dependencies.

scToolkit does **not** claim ownership of, is not derived from, and does not promote itself as an alternative or extension officially associated with the Scanpy project.  
The name “Scanpy” is used strictly in a descriptive manner under fair-use terms, and users should refer to the official Scanpy repository for the authoritative implementation.


## Installation

This version is only tested with https://github.com/conda-forge/miniforge/releases/download/23.3.1-0/Miniforge3-23.3.1-0-Linux-aarch64.sh
If it is not available anymore, please try a similar version.

> GENERAL NOTE: Make sure to replace the paths properly, if it has a '/' in the end it must be there.
> And also if absolute is in the dummy paht, use the absolut path!
0. Install mamba (conda)
   > NOTE: Solving the environment with conda won't work, use mamba please.
   ```bash
   CONDA_INSTALL_DIR="/absolute/path/where/you/save/your/conda/installations/"
   cd "$CONDA_INSTALL_DIR"
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   # Install the environment by following the instructions, Don't run the conda init, this would replaces the default.
   sh Miniforge3-23.3.1-0-Linux-aarch64.sh
   # Now the conda should be available and you can activate it with
   source "$CONDA_INSTALL_DIR/miniconda/bin/activate"
   ```

1. Create and activate a virtual environment for your project. You should use mamba for this:
   > NOTE 1: Make sure, the active environment is the conda base
   
   > NOTE 2: Make sure to have a cuda capable GPU available, 
   > if you install the GPU version.

   ```bash
   # ######################
   # CONFIGURATION VARIABLES
   GIT_DIR="/absolute/path/to/git/folder/"
   ENV_PATH="/absolute/path/to/env"
   # #######
   KERNEL_NAME="scToolkit_25_04"
   ENV_YML="env_local_25-04.yml"
   # Mac or no GPU available
   # ENV_YML="env_local_no_gpu_25-04.yml" 
   # Developers
   # ENV_YML="env_devel_25-04.yml"
   # ######################
   # EXECUTION
   # Navigate to conda environment definition
   cd "${GIT_DIR}conda" || { echo "Failed to cd to ${GIT_DIR}conda"; exit 1; }
   # Create the environment with mamba
   mamba env create -y -f "$ENV_YML" -p "$ENV_PATH"
   # Activate the environment
   conda activate "$ENV_PATH"
   echo "Environment setup complete. '${ENV_PATH}' is ready to use."
   ```

> **Note for Mac users**  
> In case you have a mac with latest M2/M3 you might need to 
> alter the environment creation. Below the example.
>
> ```bash
> ENV_YML="env_local_no_gpu_25-04.yml"
> # Create the environment with mamba
> CONDA_SUBDIR=osx-64 mamba env create -y -f "$ENV_YML" -p "$ENV_PATH"
> # Activate the environment
> mamba activate "$ENV_PATH"
> conda config --env --set subdir osx-64
> ```

2. Install scToolkit
   ```bash
   # Install the local package
   cd "$GIT_DIR" || { echo "Failed to cd to $GIT_DIR"; exit 1; }
   pip install . && rm -rf build/ src/scToolkit.egg-info
   # Add the environment to Jupyter kernels
   python -m ipykernel install --user --name "$KERNEL_NAME"
   sh setup.sh
   echo "Jupyter notebook kernel setup complete. Kernel '${KERNEL_NAME}' is ready to use."
   ```

## Optionally create the R envrionment for the R compatibility
For [Seurat](https://satijalab.org/seurat/) compatibility, some functions
require an R environment. Since certain R functions are unavailable in
Python, we provide wrappers (e.g sc_utils.convert_orthologs).

1. Create the R envrionment

   > NOTE: Make sure, the active environment is the conda base
   ```bash
   # ######################
   # CONFIGURATION VARIABLES
   GIT_DIR="/absolute/path/to/git/folder/"
   ENV_PATH="/absolute/path/to/env"
   # #######
   KERNEL_NAME="scToolkit_r_stable"
   ENV_YML="env_r.yml"
   # ######################
   # EXECUTION
   # Navigate to conda environment definition
   cd "${GIT_DIR}conda" || { echo "Failed to cd to ${GIT_DIR}conda"; exit 1; }
   # Create the environment with mamba
   mamba env create -y -f "$ENV_YML" -p "$ENV_PATH"
   # Activate the environment
   conda activate "$ENV_PATH"
   # Install seurat and create a Jupyter R kernel
   Rscript install_seurat.R 
   echo "Environment setup complete. '${ENV_PATH}' is ready to use."
   ```
2. Set the R_HOME system variable permanent or termorary

   The rpy2 needs the R_HOME variable to be set.
   
   * Set it in the bash for a permanent solution NOTE: It will be your 
      default R environment
      ```bash
      ENV_PATH="/absolute/path/to/env"
      echo "export R_HOME=\"$ENV_PATH/bin/R\"" >> ~/.bashrc
      ```
   * For a non invasive solution you can set it in python via:
      ```python
      # Inside a jupyter notebook
      %env R_HOME="/absolute/path/to/env/bin/R"
      # Inside a .py file
      import os 
      os.environ["R_HOME"] = "/absolute/path/to/env/bin/R"
      ```

## Usage
Please check the documentation in [docs](docs/html/index.html)

## Citation
Manuscript in prep.

## References
- For the docu We used the tempalte from: 
  [minimalsphinx](https://github.com/melissawm/minimalsphinx.git), [Sphinx documentation](https://www.sphinx-doc.org/en/master/)
- [scanpy](https://scanpy.readthedocs.io/en/stable/)
- [rapids_singlecell](https://github.com/scverse/rapids_singlecell)
- [CellMarker 2.0](http://biocc.hrbmu.edu.cn/CellMarker/),
  [CellTypist](https://www.celltypist.org/),
  [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb) and 
  [SC-Type](https://github.com/IanevskiAleksandr/sc-type) – integrated resources  

