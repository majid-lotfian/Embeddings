import matplotlib.pyplot as plt
import random
import numpy as np

# Create varied timestamps
timestamps = ['2024-09-01 08:00', '2024-09-01 10:00', '2024-09-03 12:00', '2024-09-10 14:00', '2024-09-25 16:00']

# Define modalities and random data assignment for the patient
modalities = ['Heart - Text', 'Heart - Image', 'Heart - Tabular', 'Blood Test']
patient_data = {timestamp: random.sample(modalities, random.randint(1, 4)) for timestamp in timestamps}

# Define modality colors
modality_colors = {
    'Heart - Text': 'red',
    'Heart - Image': 'blue',
    'Heart - Tabular': 'green',
    'Blood Test': 'purple'
}

# Initialize 2D plot for a single patient
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting modalities and merged embeddings
plotted_modalities = set()
x_offset = 0.5  # Offset to shift the first point slightly to the right

for i, timestamp in enumerate(timestamps):
    x = i 
    y_modalities = patient_data[timestamp]

    # Collect y-values for each modality to calculate merged points
    y_values = []

    for modality in y_modalities:
        y_val = random.uniform(-0.5, 0.5)
        y_values.append(y_val)
        ax.scatter(x, y_val, c=modality_colors[modality], s=150, marker='o',
                   label=modality if modality not in plotted_modalities else "")
        plotted_modalities.add(modality)

    # Plot the merged embedding exactly on the X-axis (bottom line)
    if y_values:
        ax.scatter(x, -0.8, c='black', s=200, marker='o', label='Merged Embedding' if i == 0 else "")



# Set ticks and labels
ax.set_xticks([i  for i in range(len(timestamps))])
ax.set_xticklabels(timestamps, rotation=45, ha='right')
ax.set_yticks([])  # Removing Y-axis categories
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Embeddings for Various Modalities', fontsize=12)
ax.set_title('Patient Embeddings History Over Time', fontsize=14)
ax.grid(axis='x')  # Only vertical lines for time



plt.legend(loc='upper left',bbox_to_anchor=(1.05, 1), title="Modalities")
plt.tight_layout()
plt.show()
