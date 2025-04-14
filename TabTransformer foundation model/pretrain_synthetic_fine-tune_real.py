import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tab_transformer_pytorch import TabTransformer
import torch.nn as nn
import torch.optim as optim

# Step 1: Pretraining on the Synthetic Dataset

# Load your synthetic dataset
synthetic_data_path = "synthetic_blood_count_dataset.xlsx"
synthetic_data = pd.read_excel(synthetic_data_path)

# Separate features into categorical and numerical
categorical_columns = ['Gender']  # Adjust based on your dataset's categorical columns
numerical_columns = [col for col in synthetic_data.columns if col not in categorical_columns]

# Encode categorical features
for col in categorical_columns:
    synthetic_data[col] = LabelEncoder().fit_transform(synthetic_data[col])

# Standardize numerical features
scaler = StandardScaler()
synthetic_data[numerical_columns] = scaler.fit_transform(synthetic_data[numerical_columns])

# Convert data to PyTorch tensors
X_categ_synthetic = torch.tensor(synthetic_data[categorical_columns].values, dtype=torch.long)
X_cont_synthetic = torch.tensor(synthetic_data[numerical_columns].values, dtype=torch.float32)

# Custom Dataset Class for Masked Column Prediction
class MaskedTabularDataset(Dataset):
    def __init__(self, X_categ, X_cont, mask_ratio=0.10):
        self.X_categ = X_categ
        self.X_cont = X_cont
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.X_categ)

    def __getitem__(self, idx):
        sample_categ = self.X_categ[idx].clone()
        sample_cont = self.X_cont[idx].clone()

        # Mask some of the continuous values
        mask = torch.rand(sample_cont.shape) < self.mask_ratio
        masked_sample_cont = sample_cont.clone()
        masked_sample_cont[mask] = 0  # Replace masked values with zero

        return sample_categ, masked_sample_cont, sample_cont, mask

# Initialize Dataset and DataLoader for Pretraining
pretrain_dataset = MaskedTabularDataset(X_categ_synthetic, X_cont_synthetic, mask_ratio=0.10)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

# Define the TabTransformer Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TabTransformer(
    categories=(2,),  # Adjust based on the number of unique values in categorical columns
    num_continuous=len(numerical_columns),
    dim=64,  # Embedding dimension
    dim_out=1,  # Output dimension; adjust if needed for different tasks
    depth=8,  # Number of transformer layers
    heads=8,  # Number of attention heads
    attn_dropout=0.1,
    ff_dropout=0.1
).to(device)

# Define the masked loss function
def masked_loss(predictions, targets, mask):
    predictions = predictions.squeeze()
    mask = mask.bool()
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1).repeat(1, targets.size(1))
    
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]

    if masked_predictions.numel() == 0:
        return torch.tensor(0.0, requires_grad=True).to(predictions.device)
    else:
        num_loss = nn.MSELoss()(masked_predictions, masked_targets)
        return num_loss

# Pretraining the Model
optimizer = optim.Adam(model.parameters(), lr=5e-5)
num_pretrain_epochs = 100

for epoch in range(num_pretrain_epochs):
    model.train()
    total_pretrain_loss = 0
    for sample_categ, masked_sample_cont, original_cont, mask in pretrain_loader:
        sample_categ, masked_sample_cont, original_cont, mask = (
            sample_categ.to(device), masked_sample_cont.to(device),
            original_cont.to(device), mask.to(device)
        )

        optimizer.zero_grad()
        outputs = model(x_categ=sample_categ, x_cont=masked_sample_cont)
        loss = masked_loss(outputs, original_cont, mask)
        loss.backward()
        optimizer.step()
        total_pretrain_loss += loss.item()
    
    print(f"Pretrain Epoch {epoch + 1}/{num_pretrain_epochs}, Loss: {total_pretrain_loss / len(pretrain_loader):.4f}")

print("Pretraining on the synthetic dataset complete.")

# Step 2: Fine-Tuning on the Real Dataset

# Load the real dataset for fine-tuning
real_data_path = "heart_failure_clinical_records_dataset.csv"  # Replace with your real dataset path
real_data = pd.read_csv(real_data_path)

# Preprocess the real dataset
categorical_columns = ['anaemia','diabetes','high_blood_pressure','sex','smoking','DEATH_EVENT']  # Adjust based on your dataset's categorical columns
numerical_columns = [col for col in real_data.columns if col not in categorical_columns + ['Target']]  # Adjust 'Target'

# Encode categorical features
for col in categorical_columns:
    real_data[col] = LabelEncoder().fit_transform(real_data[col])

# Standardize numerical features
real_data[numerical_columns] = scaler.transform(real_data[numerical_columns])  # Use the same scaler from pretraining

# Prepare tensors for fine-tuning
X_categ_real = torch.tensor(real_data[categorical_columns].values, dtype=torch.long)
X_cont_real = torch.tensor(real_data[numerical_columns].values, dtype=torch.float32)
y_real = torch.tensor(real_data['DEATH_EVENT'].values, dtype=torch.float32).unsqueeze(1)  # Adjust 'Target' based on your column name

# Create DataLoader for fine-tuning
fine_tune_dataset = TensorDataset(X_categ_real, X_cont_real, y_real)
fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=32, shuffle=True)

# Fine-Tuning the Model on the Real Dataset
criterion = nn.BCEWithLogitsLoss()  # Change the loss function if necessary
optimizer = optim.Adam(model.parameters(), lr=5e-5)

num_fine_tune_epochs = 100
for epoch in range(num_fine_tune_epochs):
    model.train()
    total_loss = 0
    for sample_categ, sample_cont, targets in fine_tune_loader:
        sample_categ, sample_cont, targets = sample_categ.to(device), sample_cont.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_categ=sample_categ, x_cont=sample_cont)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_fine_tune_epochs}, Loss: {total_loss / len(fine_tune_loader):.4f}")

print("Fine-tuning on the real dataset complete.")
