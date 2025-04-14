import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import plotly.graph_objects as go
from transformers import AutoModel, AutoTokenizer

# Step 1: Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Step 2: Preprocess the data
# Fill missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data.loc[:, col] = data[col].fillna(data[col].mode()[0])
    else:
        data.loc[:, col] = data[col].fillna(data[col].median())

# Separate features and labels
target = 'DEATH_EVENT'  # Modify with the actual target column name
X = data.drop(columns=[target])  # Features
y = data[target]  # Labels

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to torch tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Step 3: Load a pre-trained transformer model
# You can choose a suitable model like "microsoft/tabular-transformer" if available, here we use a generic example
model_name = "bert-base-uncased"  # Example model, replace with a tabular-specific model if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Step 4: Generate embeddings
def generate_transformer_embeddings(model, X, device):
    with torch.no_grad():
        X = X.to(device)
        # Forward pass to generate embeddings from the encoder
        embeddings = model(X)[0].cpu().numpy()  # Adjust indexing based on model output
    return embeddings

# Generate high-dimensional embeddings
high_dim_embeddings = generate_transformer_embeddings(model, X_tensor, device)

# Save the embeddings
np.save('./embeddings/high_dim_embeddings.npy', high_dim_embeddings)
print(f"Generated high-dimensional embeddings shape: {high_dim_embeddings.shape}")

# Step 5: Load saved embeddings for visualization
loaded_embeddings = np.load('./embeddings/high_dim_embeddings.npy')

# Step 6: Apply PCA to reduce dimensionality
n_components = min(loaded_embeddings.shape[0], loaded_embeddings.shape[1], 50)  # Ensure n_components is within allowed range
print(f"Using {n_components} components for PCA.")
pca = PCA(n_components=n_components, random_state=42)
embeddings_pca = pca.fit_transform(loaded_embeddings)

# Step 7: Apply t-SNE on the PCA-reduced embeddings
tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=3000, metric='cosine')
embeddings_tsne = tsne.fit_transform(embeddings_pca)

# Step 8: Visualize the t-SNE results in 3D using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=embeddings_tsne[:, 0],
    y=embeddings_tsne[:, 1],
    z=embeddings_tsne[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        opacity=0.8
    )
)])

fig.update_layout(
    title='3D t-SNE Visualization of Embeddings (PCA + t-SNE)',
    scene=dict(
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        zaxis_title='t-SNE Dimension 3'
    )
)

# Save the figure as an interactive HTML file
fig.write_html("./plot/tsne_pca_embeddings_3D_plot.html")
fig.show()

print("Embeddings generated, visualized, and saved successfully as an HTML file.")
