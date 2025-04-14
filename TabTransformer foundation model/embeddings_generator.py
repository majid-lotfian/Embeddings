import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tab_transformer_pytorch import TabTransformer
from torch.utils.data import DataLoader, Dataset

# Load your synthetic dataset
synthetic_data_path = "synthetic_blood_count_dataset.xlsx"  # Update with your actual path
synthetic_data = pd.read_excel(synthetic_data_path)

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

# Define the TabTransformer model (ensure this matches the model used for pretraining)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TabTransformer(
    categories=(2,),  # Adjust based on unique values in categorical columns
    num_continuous=len(numerical_columns),
    dim=64,  # Embedding dimension
    dim_out=1,  # Output dimension; usually used for classification/regression
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

# Save embeddings as a numpy file for future use
np.save('synthetic_embeddings.npy', embeddings_array)

print("Embeddings generated and saved successfully.")
