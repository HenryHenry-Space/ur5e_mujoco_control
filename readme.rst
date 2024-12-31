MuJoCo Environment Setup
========================

This repository contains scripts for setting up a MuJoCo simulation environment.

Requirements
------------
- Conda (Miniconda or Anaconda)
- Bash shell

Setup Instructions
------------------

1. Make the setup script executable::

   chmod +x setup_mujoco_env.sh

2. Run the setup script::

   ./setup_mujoco_env.sh

3. Activate the environment::

   conda activate mujoco_env

Installed Packages
------------------
- Python 3.9
- MuJoCo
- NumPy
- SciPy
- Matplotlib
- numpy-quaternion (for quaternion operations)
- transforms3d (for transformation operations)

Testing the Installation
------------------------

After activation, you can test the installation by running::

   python -c "import mujoco; print(mujoco.__version__)"

Inverse Kinematics
-------------------

The ``ur5e_inverseKInematics.py`` script demonstrates inverse kinematics control for the UR5e robotic arm using MuJoCo simulation.

Key Features
~~~~~~~~~~~~
- End-effector trajectory tracking
- Interactive 3D visualization

Usage
~~~~~

1. Install dependencies::

   pip install mujoco numpy plotly

2. Run the controller::

   python ur5e_inverseKInematics.py

iLQG Control
------------

The ``ur5e_ilqg.py`` script demonstrates iterative Linear Quadratic Gaussian (iLQG) control for the UR5e robotic arm using MuJoCo simulation.

System Dynamics
~~~~~~~~~~~~~~~

The system uses a linear approximation::

   \dot{x} = f(x,u) \approx f_x(x-x_0) + f_u(u-u_0)

State transition matrices::

   f_x = \begin{bmatrix} 
   0_{6\times6} & I_{6\times6} \\
   0_{6\times6} & 0_{6\times6}
   \end{bmatrix} \Delta t

   f_u = \begin{bmatrix}
   0_{6\times6} \\
   I_{6\times6}
   \end{bmatrix} \Delta t

Cost Function
~~~~~~~~~~~~~

::

   J = x_N^T Q_f x_N + \sum_{t=0}^{N-1} (x_t^T Q x_t + u_t^T R u_t)

Parameters:
- Q = diag([100,100,100,10,10,10,1,1,1,1,1,1])
- R = 0.01 I₆ₓ₆
- Qf = 5Q

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

Backward Pass
^^^^^^^^^^^^^

Value function approximation::

   V(x_t) \approx \frac{1}{2}x_t^T V_{xx} x_t + V_x^T x_t

Gains computation::

   k_t = -Q_{uu}^{-1} Q_u
   K_t = -Q_{uu}^{-1} Q_{ux}

Forward Pass
^^^^^^^^^^^^

Control law::

   u_t = u_t^{old} + \alpha k_t + K_t(x_t - x_t^{old})

Usage
~~~~~

1. Install dependencies::

   pip install mujoco numpy plotly

2. Run the controller::

   python ur5e_ilqg.py

Key Features
~~~~~~~~~~~~
- End-effector trajectory tracking
- Collision avoidance with base table
- Interactive 3D visualization
- Multiple initial seeds for local minima escape

Parameters
~~~~~~~~~~
- State dimension (nx): 12 (6 joint positions + 6 velocities)
- Control dimension (nu): 6 (joint torques)
- Time step (dt): 0.01s
- Control horizon: 30 steps
- Maximum iterations: 100

UR5e XML Configuration
----------------------

The ``ur5e.xml`` file contains the MuJoCo model configuration for the UR5e robotic arm.

Key Components
~~~~~~~~~~~~~~

- **Joints**: Defines the joints of the robotic arm.
  Example::

    <joint name="shoulder_pan_joint" class="size3" axis="0 0 1"/>

- **Geometries**: Defines the visual and collision geometries.
  Example::

    <geom mesh="shoulder_0" material="urblue" class="visual"/>
    <geom class="collision" size="0.06 0.06" pos="0 0 -0.04"/>

- **Bodies**: Defines the links of the robotic arm.
  Example::

    <body name="upper_arm_link" pos="0 0.138 0" quat="1 0 1 0">
        <inertial mass="8.393" pos="0 0 0.2125" diaginertia="0.133886 0.133886 0.0151074"/>
        <joint name="shoulder_lift_joint" class="size3"/>
        <geom mesh="upperarm_0" material="linkgray" class="visual"/>
        <geom class="collision" pos="0 -0.04 0" quat="1 1 0 0" size="0.06 0.06"/>
        <geom class="collision" size="0.05 0.2" pos="0 0 0.2"/>

- **Inertial Properties**: Defines the mass and inertia of the links.
  Example::

    <inertial mass="8.393" pos="0 0 0.2125" diaginertia="0.133886 0.133886 0.0151074"/>

- **Hierarchy**: Defines the hierarchical structure of the robotic arm.
  Example::

    <body name="forearm_link" pos="0 -0.131 0.425">
        <inertial mass="2.275" pos="0 0 0.196" diaginertia="0.0311796 0.0311796 0.004095"/>
        <joint name="elbow_joint" class="size3_limited"/>
        <geom mesh="forearm_0" material="urblue" class="visual"/>
        <geom class="collision" pos="0 0.08 0" quat="1 1 0 0" size="0.055 0.06"/>
        <geom class="collision" size="0.038 0.19" pos="0 0 0.2"/>

        <body name="wrist_1_link" pos="0 0 0.392" quat="1 0 1 0">
            <inertial mass="1.219" pos="0 0.127 0" diaginertia="0.0025599 0.0025599 0.0021942"/>
            <joint name="wrist_1_joint" class="size1"/>
            <geom mesh="wrist1_0" material="black" class="visual"/>
            <geom class="collision" pos="0 0.05 0" quat="1 1 0 0" size="0.04 0.07"/>

            <body name="wrist_2_link" pos="0 0.127 0">
                <inertial mass="1.219" pos="0 0 0.1" diaginertia="0.0025599 0.0025599 0.0021942"/>
                <joint name="wrist_2_joint" axis="0 0 1" class="size1"/>
                <geom mesh="wrist2_0" material="black" class="visual"/>
                <geom class="collision" size="0.04 0.06" pos="0 0 0.04"/>
                <geom class="collision" pos="0 0.02 0.1" quat="1 1 0 0" size="0.04 0.04"/>

                <body name="wrist_3_link" pos="0 0 0.1">
                    <inertial mass="0.1889" pos="0 0.0771683 0" quat="1 0 0 1"
                        diaginertia="0.000132134 9.90863e-05 9.90863e-05"/>
                    <joint name="wrist_3_joint" class="size1"/>