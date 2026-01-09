# IK Dataset Generation (ABB IRB4600 / IRB1520D)

This repository contains notebook-based code to generate supervised datasets for **inverse kinematics (IK)** of ABB 6‑DOF manipulators using synthetic sinusoidal joint trajectories.  
Each dataset row contains the end‑effector pose in Cartesian space and the corresponding joint angles, ready to train neural IK models such as ANN or ESRNN.

---

## 🔧 What this repo does

The notebooks:

- Generate smooth joint trajectories for all 6 joints using parameterized sinusoidal functions that respect typical ABB joint limits.
- Compute forward kinematics (FK) using Denavit–Hartenberg (DH) parameters for the ABB IRB4600 / IRB1520D arm.  
- Convert the rotation matrix to Euler angles (XYZ convention, in degrees) for orientation representation.
- Concatenate position, orientation, and joint angles into a single array and export it as a CSV file for IK learning experiments.

This gives a large paired dataset \((x, y)\), where:

- **Input**: End‑effector pose \([x, y, z, \text{roll}, \text{pitch}, \text{yaw}]\)  
- **Output**: Joint angles \([j_1, j_2, j_3, j_4, j_5, j_6]\)

---

## 📁 Notebooks and outputs

### `sinusoidal_ABB.ipynb`

Generates a **time‑series sinusoidal dataset** for a 6‑DOF ABB arm.

- Samples `n_samples = 50_000` time steps over one period:  
  \[
  t \in [0, 2\pi]
  \]
  using `np.linspace(0, 2*np.pi, n_samples)`.  
- Defines six sinusoidal joint trajectories (in degrees) such as:  
  ```python
  θ1 = 170 * np.sin(t)                       # approx [-170, 170]
  θ2 = 80 * np.sin(2*t + np.pi/6)           # approx [-75, 85]
  θ3 = 120 * np.sin(1.5*t + np.pi/3) - 52.5 # approx [-170, 70]
  θ4 = 200 * np.sin(2.5*t + np.pi/2)        # approx [-200, 200]
  θ5 = 120 * np.sin(t + np.pi/4) - 2.5      # approx [-120, 120]
  θ6 = 400 * np.sin(3*t + np.pi/5)          # approx [-400, 400]
  ```
- Implements FK with DH parameters and builds the homogeneous transform:

  ```python
  def dh_matrix(theta, d, a, alpha):
      ct, st = np.cos(theta), np.sin(theta)
      ca, sa = np.cos(alpha), np.sin(alpha)
      return np.array([
          [ct, -st*ca,  st*sa,  a*ct],
          [st,  ct*ca, -ct*sa,  a*st],
          [0,       sa,     ca,    d],
         ,[3]
      ])

  def fk_irb4600(joint_angles_deg):
      theta = np.radians(joint_angles_deg)
      dh_params = [
          [theta, 0,   160,  np.pi/2],
          [theta, 0,   590,  0],[3]
          [theta, 0,   200,  np.pi/2],[4]
          [theta, 723, 0,    np.pi/2],[5]
          [theta, 0,   0,   -np.pi/2],[6]
          [theta, 200, 0,    0],[7]
      ]
      T = np.eye(4)
      for p in dh_params:
          T = T @ dh_matrix(*p)
      position = T[:3, 3]
      rotation = T[:3, :3]
      return position, rotation
  ```

- Converts the rotation matrix to Euler angles:

  ```python
  from scipy.spatial.transform import Rotation as R

  def rotation_matrix_to_euler(R_mat):
      r = R.from_matrix(R_mat)
      euler_angles = r.as_euler('xyz', degrees=True)
      return euler_angles  # [roll(rx), pitch(ry), yaw(rz)]
  ```

- For each time step, concatenates pose and angles:

  ```python
  dataset = []
  for i in range(len(θ1)):
      angles = [θ1[i], θ2[i], θ3[i], θ4[i], θ5[i], θ6[i]]
      pos, rot = fk_irb4600(angles)
      euler_angles = rotation_matrix_to_euler(rot)
      res = np.concatenate((pos, euler_angles, angles))
      dataset.append(res)
  ```

- Saves the dataset as:

  ```python
  columns = ['x', 'y', 'z', 'roll(rx)', 'pitch(ry)', 'yaw(rz)',
             'j1', 'j2', 'j3', 'j4', 'j5', 'j6']

  df = pd.DataFrame(dataset, columns=columns)
  df.to_csv('IK_sinusoidal_dataset_irb1520D.csv', index=False)
  ```

> Note: The file name currently uses `irb1520D` but the DH parameters match the IRB4600 block above; adjust the name or DH parameters if you want to clearly separate robots.

### `sinusoidal_irb1520_cartesian.ipynb`

Generates a **Cartesian‑space enriched dataset** by combining multiple discrete joint samples via a Cartesian product.

- Uses `n_samples = 8` sinusoidal samples per joint:

  ```python
  n_samples = 8
  t = np.linspace(0, 2*np.pi, n_samples)

  θ1 = 170 * np.sin(t)
  θ2 = 80  * np.sin(2*t + np.pi/6)
  θ3 = 120 * np.sin(1.5*t + np.pi/3) - 52.5
  θ4 = 200 * np.sin(2.5*t + np.pi/2)
  θ5 = 120 * np.sin(t + np.pi/4) - 2.5
  θ6 = 400 * np.sin(3*t + np.pi/5)
  ``` 

- Builds the full grid of joint combinations:

  ```python
  import itertools

  cartesian_product = list(itertools.product(θ1, θ2, θ3, θ4, θ5, θ6))
  dataset = np.array(cartesian_product)
  print("Shape of dataset:", dataset.shape)  # (262144, 6)
  ``` 

- Reuses the same FK and Euler conversion functions and processes all 262,144 samples:

  ```python
  detset = []
  for i in range(dataset.shape):
      angles = dataset[i]
      pos, rot = fk_irb4600(angles)
      euler_angles = rotation_matrix_to_euler(rot)
      res = np.concatenate((pos, euler_angles, angles))
      detset.append(res)
  ```

- Saves the result:

  ```python
  columns = ['x', 'y', 'z', 'roll(rx)', 'pitch(ry)', 'yaw(rz)',
             'j1', 'j2', 'j3', 'j4', 'j5', 'j6']

  df = pd.DataFrame(detset, columns=columns)
  df.to_csv('IK_sinusoidal_irb1520D_262000.csv', index=False)
  ``` 

This notebook is useful when you want a dense coverage of the joint space through a Cartesian product of discretized joint samples.

---

## 🚀 How to run

1. **Clone the repository**

   ```bash
   git clone https://github.com/adarshkatare6/ik_dataset_generation.git
   cd ik_dataset_generation
