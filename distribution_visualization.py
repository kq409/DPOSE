import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
data = np.loadtxt('../diffusion_dataset/camera_positions.txt',
                  delimiter=',')
df = pd.DataFrame(data, columns=['x', 'y', 'z'])

duplicates_count = df.groupby(['x', 'y']).size().reset_index(name='count')

# Visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(duplicates_count['x'], duplicates_count['y'],
                      c=duplicates_count['count'],
                      cmap='viridis', alpha=0.7)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Camera XY Duplicate')
plt.colorbar(scatter, label='Number of Duplicates')
plt.grid(True)
plt.show()
