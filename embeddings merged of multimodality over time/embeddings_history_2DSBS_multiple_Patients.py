import matplotlib.pyplot as plt
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

# Initialize 3D-like plot with multiple 2D subplots side by side
fig, axes = plt.subplots(1, num_patients, figsize=(15, 6), sharey=True)

# Iterate over each patient to plot their embedding history
for z, (patient_id, patient_data) in enumerate(all_patient_data.items()):
    ax = axes[z]
    plotted_modalities = set()

    # Plotting modalities and merged embeddings for each timestamp
    for i, timestamp in enumerate(timestamps):
        x = i  # X-axis corresponds to timestamps
        y_modalities = patient_data[timestamp]
        y_values = []

        # Plot individual modality points
        for modality in y_modalities:
            y_val = random.uniform(-0.5, 0.5)
            y_values.append(y_val)
            ax.scatter(x, y_val, c=modality_colors[modality], s=150, marker='o',
                       label=modality if modality not in plotted_modalities else "")
            plotted_modalities.add(modality)

        # Plot the merged embedding exactly on the X-axis (bottom line)
        if y_values:
            ax.scatter(x, -0.8, c='black', s=200, marker='o', label='Merged Embedding' if i == 0 else "")

    # Set ticks, labels, and title for each subplot
    ax.set_xticks([i for i in range(len(timestamps))])
    ax.set_xticklabels(timestamps, rotation=45, ha='right')
    ax.set_yticks([])  # Removing Y-axis categories
    ax.set_xlabel('Time', fontsize=10)
    ax.set_title(f'{patient_id}', fontsize=12)
    ax.grid(axis='x')  # Only vertical lines for time

# Set common Y-axis label
fig.supylabel('Embeddings for Various Modalities', fontsize=12)
fig.suptitle('Embedding Histories for Multiple Patients', fontsize=14)

# Adding the legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Modalities")
plt.tight_layout()
plt.show()