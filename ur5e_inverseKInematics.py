import mujoco
import numpy as np
from mujoco import viewer
import time
import plotly.graph_objects as go

class UR5eController:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = viewer.launch_passive(self.model, self.data)
        
        self.dt = 0.005
        self.max_steps = 5000
        self.tolerance = 0.1
        self.damping = 0.5
        self.trajectories = []  # Store all trajectories
        
    def get_ee_pos(self):
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        return self.data.site_xpos[ee_site_id].copy()
    
    def move_ee(self, target_pos, speed=2.0):
        print(f"Moving to target: {target_pos}")
        trajectory = []
        
        for step in range(self.max_steps):
            current_pos = self.get_ee_pos()
            trajectory.append(current_pos)
            
            error = target_pos - current_pos
            distance = np.linalg.norm(error)
            
            if distance < self.tolerance:
                print(f"Target reached in {step} steps!")
                print("End effector coordinates:")
                print(f"Start: {trajectory[0]}")
                print(f"End: {trajectory[-1]}")
                self.trajectories.append(np.array(trajectory))
                return True
            
            jacp = np.zeros((3, self.model.nv))
            ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            mujoco.mj_jacSite(self.model, self.data, jacp, None, ee_site_id)
            
            J = jacp[:, :6]
            J_T = J.T
            lambda_square = self.damping ** 2
            J_pinv = J_T @ np.linalg.inv(J @ J_T + lambda_square * np.eye(3))
            
            scale = min(1.0, distance)
            qvel = J_pinv @ (error * speed * scale)
            qvel = np.clip(qvel, -2.0, 2.0)
            
            self.data.qvel[:6] = qvel
            mujoco.mj_step(self.model, self.data)
            
            self.viewer.sync()
            time.sleep(self.dt*0.001)
            
            if step % 100 == 0:
                print(f"Step {step}, Distance: {distance:.4f}")
        
        self.trajectories.append(np.array(trajectory))
        return False
    
    def plot_trajectories(self):
        fig = go.Figure()
        
        colors = ['blue', 'red']
        names = ['A to Start', 'Start to B']
        
        for traj, color, name in zip(self.trajectories, colors, names):
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))
            
            # Add start and end points
            fig.add_trace(go.Scatter3d(
                x=[traj[0, 0]],
                y=[traj[0, 1]],
                z=[traj[0, 2]],
                mode='markers',
                name=f'{name} Start',
                marker=dict(size=8, color=color)
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[traj[-1, 0]],
                y=[traj[-1, 1]],
                z=[traj[-1, 2]],
                mode='markers',
                name=f'{name} End',
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
            width=800,
            height=800
        )
        
        fig.write_html("ur5e_trajectories.html")
        print("Trajectory plot saved as ur5e_trajectories.html")
    
    def close(self):
        self.viewer.close()

def main():
    try:
        controller = UR5eController("ur5e.xml")
        
        point_a = np.array([0.3, 0.3, 1.0])
        point_b = np.array([0.3, -0.3, 1.0])
        
        print("\nMoving to point A...")
        controller.move_ee(point_a)
        time.sleep(0.5)
        
        print("\nMoving to point B...")
        controller.move_ee(point_b)
        
        # Generate interactive plot
        controller.plot_trajectories()
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'controller' in locals():
            controller.close()

if __name__ == "__main__":
    main()