import mujoco
import numpy as np
from mujoco import viewer
import time
import plotly.graph_objects as go

class UR5eiLQG:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = viewer.launch_passive(self.model, self.data)
        
        self.nx = 12
        self.nu = 6
        self.dt = 0.01
        self.horizon = 30
        self.max_iter = 100
        self.reg_min = 1e-6
        self.reg_max = 1e10
        self.reg_factor = 10
        self.reg = 1.0
        
        self.Q = np.diag([100.0] * 3 + [10.0] * 3 + [1.0] * 6)  # Higher weights for position
        self.R = np.diag([0.01] * self.nu)
        self.Qf = self.Q * 5
        self.trajectories = []
        
    def get_ee_pos(self):
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        return self.data.site_xpos[ee_site_id].copy()
        
    def get_state(self):
        return np.concatenate([self.data.qpos[:6], self.data.qvel[:6]])
    
    def set_state(self, x):
        self.data.qpos[:6] = x[:6]
        self.data.qvel[:6] = x[6:]
        mujoco.mj_forward(self.model, self.data)
        
    def step(self, u):
        self.data.ctrl[:6] = np.clip(u, -10, 10)
        mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def move_to_target_pos(self, target_pos):
        trajectory = []
        x0 = self.get_state()
        u_seq = np.zeros((self.horizon, self.nu))
        
        for iteration in range(self.max_iter):
            x_seq = np.zeros((self.horizon + 1, self.nx))
            x_seq[0] = x0
            ee_pos = self.get_ee_pos()
            trajectory.append(ee_pos)
            
            # Check position error
            pos_error = np.linalg.norm(target_pos - ee_pos)
            if pos_error < 0.01:
                print(f"Reached target position at iteration {iteration}")
                self.trajectories.append(np.array(trajectory))
                return True
            
            # Forward pass
            for t in range(self.horizon):
                x_seq[t + 1] = self.step(u_seq[t])
                self.set_state(x_seq[t])
                
            # Backward pass logic remains the same as before
            k_seq = np.zeros((self.horizon, self.nu))
            K_seq = np.zeros((self.horizon, self.nu, self.nx))
            
            success = False
            while not success and self.reg < self.reg_max:
                vx = self.Qf @ (x_seq[-1] - x0)  # Use initial state as reference
                vxx = self.Qf + self.reg * np.eye(self.nx)
                
                for t in range(self.horizon - 1, -1, -1):
                    state_error = x_seq[t] - x0
                    Qx = self.Q @ state_error * self.dt
                    Qu = self.R @ u_seq[t] * self.dt
                    Qxx = self.Q * self.dt
                    Quu = self.R * self.dt + self.reg * np.eye(self.nu)
                    
                    fx = np.eye(self.nx) + np.vstack([
                        np.zeros((6, 12)),
                        np.hstack([np.zeros((6, 6)), np.eye(6)])
                    ]) * self.dt
                    
                    fu = np.vstack([
                        np.zeros((6, 6)),
                        np.eye(6)
                    ]) * self.dt

                    try:
                        H = Quu + fu.T @ vxx @ fu
                        g = Qu + fu.T @ vx
                        k = -np.linalg.solve(H, g)
                        K = -np.linalg.solve(H, fu.T @ vxx @ fx)
                        
                        k_seq[t] = k
                        K_seq[t] = K
                        vx = Qx + fx.T @ vx + K.T @ H @ k
                        vxx = Qxx + fx.T @ vxx @ fx + K.T @ H @ K
                        success = True
                        
                    except np.linalg.LinAlgError:
                        success = False
                        self.reg *= self.reg_factor
                        break
            
            if not success:
                continue
                
            alpha = 1.0
            x_new = x0
            u_new = u_seq.copy()
            
            for t in range(self.horizon):
                du = k_seq[t] + K_seq[t] @ (x_new - x_seq[t])
                u_new[t] = u_seq[t] + alpha * du
                x_new = self.step(u_new[t])
                self.viewer.sync()
                time.sleep(self.dt)
            
            u_seq = u_new
            self.reg = max(self.reg_min, self.reg / self.reg_factor)
            
            ee_pos = self.get_ee_pos()
            error = np.linalg.norm(target_pos - ee_pos)
            print(f"Iteration {iteration}, Error: {error:.4f}, Reg: {self.reg:.6f}")
        
        self.trajectories.append(np.array(trajectory))
        return False

    def plot_trajectories(self):
        fig = go.Figure()
        colors = ['blue', 'red']
        names = ['A to Start', 'Start to B']
        
        for traj, color, name in zip(self.trajectories, colors, names):
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode='lines', name=name, line=dict(color=color, width=2)
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
                mode='markers', name=f'{name} Start',
                marker=dict(size=8, color=color)
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
                mode='markers', name=f'{name} End',
                marker=dict(size=8, color=color, symbol='diamond')
            ))
        
        fig.update_layout(
            title='End Effector Trajectories',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=800, height=800
        )
        
        fig.write_html("ur5e_ilqg_trajectories.html")
        print("Trajectory plot saved as ur5e_ilqg_trajectories.html")

    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()

def main():
    try:
        controller = UR5eiLQG("ur5e.xml")
        
        point_a = np.array([10.3, 10.3, 10.0])
        point_b = np.array([10.3, 4.3, 10.0])
        
        print("Moving to point A...")
        controller.move_to_target_pos(point_a)
        time.sleep(0.5)
        
        print("Moving to point B...")
        controller.move_to_target_pos(point_b)
        
        controller.plot_trajectories()
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'controller' in locals():
            controller.close()

if __name__ == "__main__":
    main()