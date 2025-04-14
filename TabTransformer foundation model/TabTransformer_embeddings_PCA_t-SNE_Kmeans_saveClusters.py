import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tab_transformer_pytorch import TabTransformer
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import os

# Create directories to save plots and cluster CSVs
#os.makedirs("./plots", exist_ok=True)
#os.makedirs("./clusters", exist_ok=True)

# Load your synthetic dataset
synthetic_data_path = "./data/dataset3/blood_count_dataset.csv"  # Update with your actual path
synthetic_data = pd.read_csv(synthetic_data_path)

# Define categorical and numerical columns (exclude target columns)
categorical_columns = ['Gender']  # Update based on your dataset
numerical_columns = [col for col in synthetic_data.columns if col not in categorical_columns]

# Encode categorical features
for col in categorical_columns:
    synthetic_data[col] = LabelEncoder().fit_transform(synthetic_data[col])

# Standardize numerical features
scaler = StandardScaler()
synthetic_data[numerical_columns] = scaler.fit_transform(synthetic_data[numerical_columns])

# Convert data to PyTorch tensors
X_categ = torch.tensor(synthetic_data[categorical_columns].values, dtype=torch.long)
X_cont = torch.tensor(synthetic_data[numerical_columns].values, dtype=torch.float32)

# Define the TabTransformer model with 128-dimensional output embeddings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TabTransformer(
    categories=(2,),  # Adjust based on unique values in categorical columns
    num_continuous=len(numerical_columns),
    dim=128,  # Set embedding dimension to 128 for comparison
    dim_out=128,  # Ensure the output dimension matches for consistency
    depth=6,  # Number of transformer layers
    heads=8,  # Number of attention heads
    attn_dropout=0.1,
    ff_dropout=0.1
).to(device)

# Load pretrained model weights if available
# model.load_state_dict(torch.load('path_to_pretrained_model.pth'))

# Set model to evaluation mode
model.eval()

# Custom Dataset for extracting embeddings
class TabDataset(Dataset):
    def __init__(self, X_categ, X_cont):
        self.X_categ = X_categ
        self.X_cont = X_cont

    def __len__(self):
        return len(self.X_categ)

    def __getitem__(self, idx):
        return self.X_categ[idx], self.X_cont[idx]

# Create DataLoader
embedding_dataset = TabDataset(X_categ, X_cont)
embedding_loader = DataLoader(embedding_dataset, batch_size=32, shuffle=False)

# Extract embeddings
all_embeddings = []

with torch.no_grad():
    for batch_categ, batch_cont in embedding_loader:
        batch_categ, batch_cont = batch_categ.to(device), batch_cont.to(device)
        embeddings = model(x_categ=batch_categ, x_cont=batch_cont)
        all_embeddings.append(embeddings.cpu())

# Concatenate all embeddings into a single tensor
all_embeddings = torch.cat(all_embeddings, dim=0)

# Convert embeddings to a numpy array for further use
embeddings_array = all_embeddings.numpy()

# Save embeddings to a file for future use or comparison
np.save('./embeddings/BC_embeddings.npy', embeddings_array)

# Step 1: Apply PCA to reduce dimensionality
pca = PCA(n_components=16, random_state=42)  # Reduce to 16 dimensions to retain more variance before t-SNE
embeddings_pca = pca.fit_transform(embeddings_array)

# Step 2: Apply t-SNE to further reduce to 3 dimensions
tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=3000)
embeddings_tsne = tsne.fit_transform(embeddings_pca)

# Step 3: Apply K-Means clustering to color points
kmeans = KMeans(n_clusters=3, random_state=42)  # Set number of clusters based on analysis
labels = kmeans.fit_predict(embeddings_tsne)

# Save each cluster's data points to separate CSV files
for cluster_label in np.unique(labels):
    cluster_data = synthetic_data[labels == cluster_label]
    cluster_data.to_csv(f'./clusters/cluster_{cluster_label}.csv', index=False)

# Step 4: Visualize in 3D using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=embeddings_tsne[:, 0],
    y=embeddings_tsne[:, 1],
    z=embeddings_tsne[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=labels,  # Color by cluster labels from K-Means
        colorscale='Viridis',  # Color scale
        opacity=0.8
    )
)])

fig.update_layout(
    title='3D t-SNE Visualization of Embeddings with K-Means Clustering',
    scene=dict(
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        zaxis_title='t-SNE Dimension 3'
    )
)

# Save the figure as an interactive HTML file
fig.write_html("./plots/BC_embeddings_tsne_3D_plot.html")

# Show the plot
fig.show()

print("Embeddings generated, visualized, and saved successfully as an HTML file.")
