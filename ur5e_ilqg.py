import mujoco
import numpy as np
from mujoco import viewer
import time
import plotly.graph_objects as go

class UR5eiLQG:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        try:
            self.viewer = viewer.launch_passive(self.model, self.data)
        except Exception as e:
            print(f"Viewer launch failed: {e}. Proceeding without viewer.")
            self.viewer = None
        
        self.nq = self.model.nq  # 6
        self.nv = self.model.nv  # 6
        self.nu = self.model.nu  # 6
        self.nx = self.nq + self.nv  # 12
        self.dt = 0.01  
        self.horizon = 30
        self.max_iter = 100
        self.reg_min = 1e-6
        self.reg_max = 1e10
        self.reg_factor = 10
        self.reg = 1.0
        
        self.W_position_error_weight = np.diag([5., 5., 5.])
        self.Q = np.diag([1.0] * self.nq + [0.1] * self.nv)
        self.R = np.diag([0.01] * self.nu)
        self.Qf = self.Q * 10
        self.trajectories = []
        
    def get_ee_pos(self):
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        return self.data.site_xpos[ee_site_id].copy()
    
    def get_EE_POS_FROM_QPOS(self, qpos):
        self.data.qpos[:self.nq] = qpos[:self.nq]
        mujoco.mj_forward(self.model, self.data)
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        return self.data.site_xpos[ee_site_id].copy()
    
    def get_EE_JACOBIAN(self):
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, ee_site_id)
        return jacp
    
    def get_state(self):
        return np.concatenate([self.data.qpos[:self.nq], self.data.qvel[:self.nv]])
    
    def set_state(self, x):
        self.data.qpos[:self.nq] = x[:self.nq]
        self.data.qvel[:self.nv] = x[self.nq:]
        mujoco.mj_forward(self.model, self.data)
    
    def step(self, u):
        self.data.ctrl[:self.nu] = np.clip(u, -self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
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
            
            pos_error = np.linalg.norm(target_pos - ee_pos)
            if pos_error < 0.01:
                print(f"Reached target position at iteration {iteration}")
                self.trajectories.append(np.array(trajectory))
                return True
            
            # Forward pass
            self.set_state(x0)
            for t in range(self.horizon):
                u = u_seq[t]
                self.data.ctrl[:self.nu] = np.clip(u, -self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
                mujoco.mj_step(self.model, self.data)
                x_seq[t + 1] = self.get_state()
            
            # Backward pass
            k_seq = np.zeros((self.horizon, self.nu))
            K_seq = np.zeros((self.horizon, self.nu, self.nx))
            
            success = False
            while not success and self.reg < self.reg_max:
                # Terminal cost
                ee_pos_final = self.get_EE_POS_FROM_QPOS(x_seq[-1][:self.nq])
                pos_error_final = ee_pos_final - target_pos
                J = self.get_EE_JACOBIAN()
                v_x_ee = np.zeros(self.nx)
                v_x_ee[self.nq:] = J.T @ (self.W_position_error_weight @ pos_error_final)
                v_x = self.Qf @ (x_seq[-1] - x0) + v_x_ee
                v_xx = self.Qf + self.reg * np.eye(self.nx)
                
                for t in range(self.horizon - 1, -1, -1):
                    fx = np.eye(self.nx) + np.vstack([
                        np.zeros((self.nq, self.nx)),
                        np.hstack([np.zeros((self.nv, self.nq)), np.eye(self.nv)])
                    ]) * self.dt
                    
                    fu = np.vstack([
                        np.zeros((self.nq, self.nu)),
                        np.eye(self.nu)
                    ]) * self.dt
                    
                    ee_pos_t = self.get_EE_POS_FROM_QPOS(x_seq[t][:self.nq])
                    J = self.get_EE_JACOBIAN()
                    pos_error = ee_pos_t - target_pos
                    
                    Q_x_ee = np.zeros(self.nx)
                    Q_x_ee[self.nq:] = J.T @ (self.W_position_error_weight @ pos_error) * self.dt
                    Q_x = self.Q @ (x_seq[t] - x0) * self.dt + Q_x_ee
                    Q_xx = self.Q * self.dt + self.reg * np.eye(self.nx)
                    Q_u = self.R @ u_seq[t] * self.dt
                    Q_uu = self.R * self.dt + self.reg * np.eye(self.nu)
                    
                    try:
                        H = Q_uu + fu.T @ v_xx @ fu
                        g = Q_u + fu.T @ v_x
                        k = -np.linalg.solve(H, g)
                        K = -np.linalg.solve(H, fu.T @ v_xx @ fx)
                        
                        k_seq[t] = k
                        K_seq[t] = K
                        v_x = Q_x + fx.T @ v_x + K.T @ H @ k
                        v_xx = Q_xx + fx.T @ v_xx @ fx + K.T @ H @ K
                        success = True
                        
                    except np.linalg.LinAlgError:
                        success = False
                        self.reg *= self.reg_factor
                        break
            
            if not success:
                print(f"Failed to converge at iteration {iteration}, reg: {self.reg}")
                continue
                
            alpha = 1.0
            x_new = x0.copy()
            u_new = u_seq.copy()
            self.set_state(x0)
            
            for t in range(self.horizon):
                du = k_seq[t] + K_seq[t] @ (x_new - x_seq[t])
                u_new[t] = u_seq[t] + alpha * du
                x_new = self.step(u_new[t])
                if self.viewer:
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
            scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
            width=800, height=800
        )
        fig.write_html("ur5e_ilqg_trajectories.html")
        print("Trajectory plot saved as ur5e_ilqg_trajectories.html")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def main():
    try:
        controller = UR5eiLQG("ur5e.xml")
        
        point_a = np.array([10.5, 0.5, 4.5])
        point_b = np.array([10.5, -0.5, 4.5])
        
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