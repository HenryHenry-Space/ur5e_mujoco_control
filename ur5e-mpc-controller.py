import mujoco
import numpy as np
from mujoco import viewer
import time
import cvxopt
from cvxopt import matrix, solvers

class UR5eMPController:
    def __init__(self, model_path):
        # Constants for MuJoCo object types
        self.JOINT = 1
        self.ACTUATOR = 3
        self.SITE = 4
        self.BODY = 0
        
        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize viewer
        self.viewer = viewer.launch_passive(self.model, self.data)
        
        # Get joint and actuator IDs
        self.joint_ids = list(range(self.model.njnt))
        self.actuator_ids = list(range(self.model.nu))
        
        # Use wrist_3_link as end-effector
        self.use_body_pos = True
        self.wrist_3_body_id = self.model.nbody - 1
        
        # System dimensions
        self.n_states = 12      # 6 joint positions + 6 joint velocities
        self.n_inputs = 6       # 6 joint torques
        self.n_outputs = 3      # 3D position of end-effector
        
        # MPC parameters
        self.N = 10            # Prediction horizon
        self.dt = self.model.opt.timestep
        
        # Initialize system matrices
        self._initialize_system_matrices()
        
        # Set initial position
        self.home_pos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        self.x = np.zeros(self.n_states)
        self.x[:6] = self.home_pos
        
        # Suppress CVXOPT output
        solvers.options['show_progress'] = False
    
    def _initialize_system_matrices(self):
        # System dynamics
        A_c = np.zeros((self.n_states, self.n_states))
        A_c[:6, 6:] = np.eye(6)
        A_c[6:, 6:] = -1.0 * np.eye(6)
        self.A = np.eye(self.n_states) + self.dt * A_c
        
        # Input matrix
        B_c = np.zeros((self.n_states, self.n_inputs))
        gains = np.array([2000, 2000, 2000, 500, 500, 500])
        B_c[6:, :] = np.diag(1.0 / gains)
        self.B = self.dt * B_c
        
        # Output matrix (for end-effector position)
        self.C = np.zeros((self.n_outputs, self.n_states))
        J = self.calculate_jacobian()
        self.C[:, :6] = J[:, :6]
        
        # Weights for MPC cost function
        self.Q = np.eye(self.n_states)  # State cost
        self.Q[:6, :6] *= 100.0         # Position cost
        self.Q[6:, 6:] *= 10.0          # Velocity cost
        self.R = np.eye(self.n_inputs) * 0.01  # Control cost
        
        # Constraints
        self.u_max = np.array([150, 150, 150, 28, 28, 28])
        self.u_min = -self.u_max
        
    def get_ee_position(self):
        return self.data.xpos[self.wrist_3_body_id]
    
    def calculate_jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.wrist_3_body_id)
        return jacp
    
    def setup_qp_matrices(self, x0, x_target):
        """Set up matrices for the QP solver"""
        n_states = self.n_states
        n_inputs = self.n_inputs
        N = self.N
        
        # Construct block diagonal matrices
        Q_bar = np.kron(np.eye(N), self.Q)
        R_bar = np.kron(np.eye(N), self.R)
        
        # Construct prediction matrices
        A_bar = np.zeros((N * n_states, n_states))
        B_bar = np.zeros((N * n_states, N * n_inputs))
        
        temp_A = np.eye(n_states)
        for i in range(N):
            A_bar[i*n_states:(i+1)*n_states, :] = temp_A
            temp_A = temp_A @ self.A
        
        for i in range(N):
            temp_B = np.zeros((N * n_states, n_inputs))
            temp_A = np.eye(n_states)
            for j in range(i+1):
                temp_B[(j)*n_states:(j+1)*n_states, :] = temp_A @ self.B
                temp_A = temp_A @ self.A
            B_bar[:, i*n_inputs:(i+1)*n_inputs] = temp_B
        
        # Construct QP matrices ensuring symmetry and positive definiteness
        H = B_bar.T @ Q_bar @ B_bar + R_bar
        H = (H + H.T) / 2  # Ensure symmetry
        H = H + 1e-6 * np.eye(H.shape[0])  # Ensure positive definiteness
        
        f = B_bar.T @ Q_bar @ A_bar @ (x0 - x_target)
        
        # Input constraints
        G = np.vstack([np.eye(N * n_inputs), -np.eye(N * n_inputs)])
        h = np.concatenate([np.tile(self.u_max, N), -np.tile(self.u_min, N)])
        
        return H, f, G, h
    
    def solve_mpc(self, x0, x_target):
        """Solve the MPC optimization problem"""
        # Setup QP matrices
        H, f, G, h = self.setup_qp_matrices(x0, x_target)
        
        # Convert to CVXOPT format and ensure correct dimensions
        P = matrix(H.astype(np.double))
        q = matrix(f.astype(np.double))
        G = matrix(G.astype(np.double))
        h = matrix(h.reshape(-1, 1).astype(np.double))  # Reshape h to column vector
        
        try:
            # Solve QP
            solution = solvers.qp(P, q, G, h)
            if solution['status'] != 'optimal':
                raise ValueError('QP solver did not find optimal solution')
            
            u = np.array(solution['x']).flatten()
            return u[:self.n_inputs]
        
        except Exception as e:
            print(f"QP solver failed: {e}")
            # Fallback control: simple PD
            error = x_target - x0
            u = np.zeros(self.n_inputs)
            u[:6] = 2000 * error[:6] + 200 * error[6:]
            u = np.clip(u, self.u_min, self.u_max)
            return u
    
    def move_to_position(self, target_pos, max_steps=600):
        print(f"Moving to target position: {target_pos}")
        
        # Get target joint positions using inverse kinematics
        J = self.calculate_jacobian()
        J_pinv = np.linalg.pinv(J)
        current_pos = self.get_ee_position()
        delta_pos = target_pos - current_pos
        target_joints = self.data.qpos + J_pinv @ delta_pos
        x_target = np.concatenate([target_joints, np.zeros(6)])
        
        # Print initial error
        initial_error = np.linalg.norm(delta_pos)
        print(f"Initial position error: {initial_error:.4f}")
        
        for step in range(max_steps):
            # Get current state
            x0 = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])
            
            # Compute MPC control
            u = self.solve_mpc(x0, x_target)
            self.data.ctrl[self.actuator_ids] = u
            
            try:
                mujoco.mj_step(self.model, self.data)
            except Exception as e:
                print(f"Simulation error: {e}")
                break
            
            self.viewer.sync()
            time.sleep(self.dt)
            
            # Print progress
            if step % 100 == 0:
                current_pos = self.get_ee_position()
                error = np.linalg.norm(target_pos - current_pos)
                print(f"Step {step}: Position error = {error:.4f}")
            
            error = np.linalg.norm(target_pos - self.get_ee_position())
            if error < 0.01:
                print(f"Reached target in {step} steps (final error: {error:.4f})")
                break
            
            if step == max_steps - 1:
                print(f"Maximum steps reached. Final error: {error:.4f}")
    
    def close(self):
        self.viewer.close()

def main():
    controller = UR5eMPController("ur5e.xml")
    
    # Initialize robot to home position
    print("\nInitializing robot to home position...")
    start_pos = controller.get_ee_position()
    print(f"Starting position: {start_pos}")
    
    # Move to top-right corner
    top_right_pos = np.array([0.3, -0.3, 0.7])  # x: right, y: back, z: up
    print(f"\nMoving to top-right position: {top_right_pos}")
    controller.move_to_position(top_right_pos)
    time.sleep(2)
    
    # Move to bottom-left corner
    bottom_left_pos = np.array([-0.3, 0.3, 0.3])  # x: left, y: forward, z: down
    print(f"\nMoving to bottom-left position: {bottom_left_pos}")
    controller.move_to_position(bottom_left_pos)
    time.sleep(2)
    
    controller.close()

if __name__ == "__main__":
    main()