import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import plotly.graph_objects as go

# Step 1: Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Step 2: Preprocess the data
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Separate features and labels if a target column exists
target = 'DEATH_EVENT'  # Modify with the actual target column name
X = data.drop(columns=[target])  # Features
y = data[target]  # Labels

# Encode categorical features and target labels
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

# Encode target labels if they are categorical
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X.values, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Select the preferred device: 'cuda' for GPU or 'cpu' for CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Step 3: Train TabNet
tabnet = TabNetClassifier(
    n_d=128, n_a=128, n_steps=7,
    gamma=1.5, lambda_sparse=1e-3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='sparsemax',  
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    epsilon=1e-15,
    device_name=device.type
)

# Fit the model on training data
tabnet.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_valid, y_valid)],
    eval_name=['valid'],
    eval_metric=['accuracy'],
    max_epochs=200,
    patience=20,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Step 4: Generate embeddings for the entire dataset and save them
def generate_embeddings(model, X, device):
    model.network.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        embeddings = model.network(X_tensor)[0].cpu().numpy()  
    return embeddings

# Generate and save embeddings for the whole dataset
all_embeddings = generate_embeddings(tabnet, X.values, device)
np.save('./embeddings/all_embeddings.npy', all_embeddings)  

# Step 5: Load saved embeddings and visualize using PCA + t-SNE
loaded_embeddings = np.load('./embeddings/all_embeddings.npy')

# Step 6: Check the dimensionality of the loaded embeddings
print(f"Loaded embeddings shape: {loaded_embeddings.shape}")

# Ensure PCA does not attempt to reduce dimensions below 3
n_components_pca = min(3, loaded_embeddings.shape[1])  # Ensure we don't exceed available dimensions
if n_components_pca < 3:
    print(f"PCA will be skipped because the input dimensions are less than 3: {n_components_pca} available.")
    embeddings_pca = loaded_embeddings  # Skip PCA if not enough dimensions
else:
    print(f"Using {n_components_pca} components for PCA.")
    pca = PCA(n_components=n_components_pca, random_state=42)
    embeddings_pca = pca.fit_transform(loaded_embeddings)

# Step 7: Apply t-SNE with 3 components if PCA output is at least 3-dimensional
tsne_components = min(3, embeddings_pca.shape[1])  # Use available dimensions for t-SNE, capped at 3
tsne = TSNE(n_components=tsne_components, random_state=42, perplexity=30, n_iter=3000, metric='cosine')
embeddings_tsne = tsne.fit_transform(embeddings_pca)

# Step 8: Visualize the t-SNE results based on the number of available dimensions
if tsne_components == 3:
    # 3D visualization if 3 components are available
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
        title='3D t-SNE Visualization of Embeddings',
        scene=dict(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            zaxis_title='t-SNE Dimension 3'
        )
    )
else:
    # Fallback to 2D visualization if only 2 components are available
    fig = go.Figure(data=[go.Scatter(
        x=embeddings_tsne[:, 0],
        y=embeddings_tsne[:, 1],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',  
            opacity=0.8
        )
    )])

    fig.update_layout(
        title='2D t-SNE Visualization of Embeddings',
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2'
    )

# Save the figure as an interactive HTML file
fig.write_html("./plot/tsne_pca_embeddings_plot.html")

fig.show()

print("Embeddings generated, visualized, and saved successfully as an HTML file.")
