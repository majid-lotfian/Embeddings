import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np

# Create varied timestamps
timestamps = ['2024-09-01 08:00', '2024-09-01 10:00', '2024-09-03 12:00', '2024-09-10 14:00', '2024-09-25 16:00']

# Define modalities and random data assignment
modalities = ['Heart - Text', 'Heart - Image', 'Heart - Tabular', 'Blood Test']
modality_colors = {
    'Heart - Text': 'red',
    'Heart - Image': 'blue',
    'Heart - Tabular': 'green',
    'Blood Test': 'purple'
}

# Define multiple patients (using random IDs for simplicity)
num_patients = 5  # Number of patients
patient_ids = [f"Patient {i+1}" for i in range(num_patients)]  # Generate patient IDs

# Generate random patient data for each patient
all_patient_data = {
    patient_id: {timestamp: random.sample(modalities, random.randint(1, 4)) for timestamp in timestamps}
    for patient_id in patient_ids
}

# Initialize 3D plot
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Iterate over each patient to plot their embedding history as a 2D slice in 3D space
for z, (patient_id, patient_data) in enumerate(all_patient_data.items()):
    plotted_modalities = set()
    
    # Plotting modalities and merged embeddings for each timestamp
    for i, timestamp in enumerate(timestamps):
        x = i  # X-axis corresponds to timestamps
        y_values = []

        # Plot individual modality points
        for modality in patient_data[timestamp]:
            y_val = random.uniform(-0.5, 0.5)
            y_values.append(y_val)
            ax.scatter(x, y_val, z, c=modality_colors[modality], s=150, marker='o',
                       label=modality if modality not in plotted_modalities else "")
            plotted_modalities.add(modality)

        # Plot the merged embedding exactly on the X-axis (bottom line)
        if y_values:
            ax.scatter(x, -0.8, z, c='black', s=200, marker='o', label='Merged Embedding' if i == 0 and z == 0 else "")

# Set axis labels and title
ax.set_xticks(range(len(timestamps)))
ax.set_xticklabels(timestamps, rotation=45, ha='right')
ax.set_yticks([])  # Removing Y-axis categories
ax.set_zticks(range(len(patient_ids)))
ax.set_zticklabels(patient_ids)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Modalities', fontsize=12)
ax.set_zlabel('Patients', fontsize=12)
ax.set_title('3D Embedding Histories for Multiple Patients', fontsize=14)

# Rotate the 3D plot for better visualization
ax.view_init(elev=20, azim=45)

# Adding the legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Modalities")
plt.tight_layout()
plt.show()
