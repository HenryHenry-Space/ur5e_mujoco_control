# MuJoCo Environment Setup

This repository contains scripts for setting up a MuJoCo simulation environment.

## Requirements
- Conda (Miniconda or Anaconda)
- Bash shell

## Setup Instructions

1. Make the setup script executable:
   ```bash
   chmod +x setup_mujoco_env.sh
   ```

2. Run the setup script:
   ```bash
   ./setup_mujoco_env.sh
   ```

3. Activate the environment:
   ```bash
   conda activate mujoco_env
   ```

## Installed Packages
- Python 3.9
- MuJoCo
- NumPy
- SciPy
- Matplotlib
- numpy-quaternion (for quaternion operations)
- transforms3d (for transformation operations)

## Testing the Installation

After activation, you can test the installation by running:
```python
python -c "import mujoco; print(mujoco.__version__)"
```
