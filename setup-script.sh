#!/bin/bash

# Define environment name and Python version
ENV_NAME="mujoco_env"
PYTHON_VERSION="3.9"  # MuJoCo works well with Python 3.9

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Conda environment for MuJoCo...${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Check if the environment already exists
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo -e "${YELLOW}Environment $ENV_NAME already exists. Do you want to remove it? [y/N]${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
        conda env remove -n $ENV_NAME
    else
        echo "Please choose a different environment name."
        exit 1
    fi
fi

echo -e "${GREEN}Creating new Conda environment...${NC}"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo -e "${GREEN}Installing required packages...${NC}"

# Install essential packages
conda install -y -c conda-forge numpy
conda install -y -c conda-forge scipy
conda install -y -c conda-forge matplotlib

# Install MuJoCo
pip install mujoco

# Install additional useful packages for robotics
pip install numpy-quaternion  # for quaternion operations
pip install transforms3d     # for transformation operations

# Verify installations
echo -e "${GREEN}Verifying installations...${NC}"
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${YELLOW}To activate the environment, run: conda activate $ENV_NAME${NC}"

# Create a requirements.txt file
echo -e "${GREEN}Creating requirements.txt...${NC}"
cat > requirements.txt << EOL
mujoco
numpy
scipy
matplotlib
numpy-quaternion
transforms3d
EOL

# Create a README with setup instructions
echo -e "${GREEN}Creating README.md...${NC}"
cat > README.md << EOL
# MuJoCo Environment Setup

This repository contains scripts for setting up a MuJoCo simulation environment.

## Requirements
- Conda (Miniconda or Anaconda)
- Bash shell

## Setup Instructions

1. Make the setup script executable:
   \`\`\`bash
   chmod +x setup_mujoco_env.sh
   \`\`\`

2. Run the setup script:
   \`\`\`bash
   ./setup_mujoco_env.sh
   \`\`\`

3. Activate the environment:
   \`\`\`bash
   conda activate mujoco_env
   \`\`\`

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
\`\`\`python
python -c "import mujoco; print(mujoco.__version__)"
\`\`\`
EOL

echo -e "${GREEN}Setup complete! Check README.md for usage instructions.${NC}"
