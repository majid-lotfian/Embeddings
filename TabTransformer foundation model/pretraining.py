import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from tab_transformer_pytorch import TabTransformer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Load your synthetic dataset
data_path = "synthetic_blood_count_dataset.xlsx"
data = pd.read_excel(data_path)

# Separate features into categorical and numerical
categorical_columns = ['Gender']  # Adjust based on your dataset's categorical columns
numerical_columns = [col for col in data.columns if col not in categorical_columns]

# Encode categorical features
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Standardize numerical features
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Convert data to PyTorch tensors
X_categ = torch.tensor(data[categorical_columns].values, dtype=torch.long)  # Categorical as long type for embedding
X_cont = torch.tensor(data[numerical_columns].values, dtype=torch.float32)  # Continuous features as float type

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Step 4.1: Define TabTransformer Model
model = TabTransformer(
    categories=(2,),  # Number of unique values per categorical column, adjust based on data
    num_continuous=len(numerical_columns),  # Number of continuous (numerical) features
    dim=32,  # Dimension of embeddings
    dim_out=1,  # Output dimension (for binary classification, adjust as needed)
    depth=6,  # Depth of transformer
    heads=8,  # Number of attention heads
    attn_dropout=0.1,  # Attention dropout rate
    ff_dropout=0.1  # Feedforward layer dropout rate
).to(device)

# Step 4.2: Pretraining - Masked Column Prediction (Optional)
# Custom Dataset Class for Masked Column Prediction
class MaskedTabularDataset(Dataset):
    def __init__(self, X_categ, X_cont, mask_ratio=0.1):
        self.X_categ = X_categ
        self.X_cont = X_cont
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.X_categ)

    def __getitem__(self, idx):
        sample_categ = self.X_categ[idx].clone()  # Clone to avoid modifying the original data
        sample_cont = self.X_cont[idx].clone()

        # Mask some of the continuous values
        mask = torch.rand(sample_cont.shape) < self.mask_ratio
        masked_sample_cont = sample_cont.clone()
        masked_sample_cont[mask] = 0  # Replace masked values with zero or other placeholder

        return sample_categ, masked_sample_cont, sample_cont, mask

# Initialize Dataset and DataLoader for Pretraining
mask_ratio = 0.15  # Percentage of values to mask in each row
pretrain_dataset = MaskedTabularDataset(X_categ, X_cont, mask_ratio=mask_ratio)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

# Define a combined loss function for masked column prediction
# Corrected masked_loss function
def masked_loss(predictions, targets, mask):
    """
    Custom loss function for masked column prediction:
    Applies MSE loss for the masked continuous values.
    """
    # Ensure mask and target dimensions are aligned correctly
    predictions = predictions.squeeze()  # Adjust shape if needed for loss calculation
    mask = mask.bool()  # Convert to boolean mask if not already

    # Adjust the dimensions to match the mask
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1).repeat(1, targets.size(1))  # Expand dimensions if necessary
    
    # Select only masked values from predictions and targets
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]

    # Compute MSE loss on masked values
    if masked_predictions.numel() == 0:  # Check if there are masked values to avoid empty tensor error
        return torch.tensor(0.0, requires_grad=True).to(predictions.device)
    else:
        num_loss = nn.MSELoss()(masked_predictions, masked_targets)
        return num_loss

# Pretraining the Model
num_pretrain_epochs = 100
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(num_pretrain_epochs):
    model.train()
    total_pretrain_loss = 0
    for sample_categ, masked_sample_cont, original_cont, mask in pretrain_loader:
        sample_categ, masked_sample_cont, original_cont, mask = (
            sample_categ.to(device), masked_sample_cont.to(device),
            original_cont.to(device), mask.to(device)
        )

        optimizer.zero_grad()
        outputs = model(x_categ=sample_categ, x_cont=masked_sample_cont)  # Pass both categorical and numerical inputs
        loss = masked_loss(outputs, original_cont, mask)
        loss.backward()
        optimizer.step()
        total_pretrain_loss += loss.item()
    
    print(f"Pretrain Epoch {epoch + 1}/{num_pretrain_epochs}, Loss: {total_pretrain_loss / len(pretrain_loader):.4f}")

print("Pretraining with Masked Column Prediction complete.")



# Fine-tuning dataset preparation
targets = torch.randint(0, 2, (X_categ.shape[0], 1)).float()  # Dummy targets for illustration
fine_tune_dataset = TensorDataset(X_categ, X_cont, targets)
fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=32, shuffle=True)

# Step 5: Fine-Tuning the Model
criterion = nn.BCEWithLogitsLoss()  # Use an appropriate loss based on your downstream task

# Fine-Tuning Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for sample_categ, sample_cont, targets in fine_tune_loader:
        sample_categ, sample_cont, targets = sample_categ.to(device), sample_cont.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_categ=sample_categ, x_cont=sample_cont)  # Pass both categorical and numerical inputs
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(fine_tune_loader):.4f}")

print("Fine-tuning complete.")
