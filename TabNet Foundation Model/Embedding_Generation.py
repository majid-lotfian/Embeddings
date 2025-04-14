import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch

# Step 1: Load the dataset
# Replace 'your_dataset.csv' with the path to your CSV file
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Step 2: Preprocess the data
# Handling missing values: fill missing values with the median for numerical columns and mode for categorical columns
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Separate features and labels if a target column exists
# Replace 'target_column' with the actual name of your target column
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
# Define the TabNet model
tabnet = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, lambda_sparse=1e-3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='sparsemax',  # Alternative: 'entmax'
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    epsilon=1e-15,
    device_name=device.type  # Correctly handles device setup inside TabNet
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

# Step 4: Extract embeddings using the TabNet's representation of inputs
def extract_embeddings(model, X):
    """
    Extracts internal representations of inputs as embeddings.
    Uses TabNet's predict_proba to access learned features.
    """
    model.network.eval()
    with torch.no_grad():
        embeddings = model.network(X)
    return embeddings[0].detach().cpu().numpy()  # Extract only the relevant output for embeddings

# Convert the input data to torch tensors before extracting embeddings
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# Extract embeddings for train, validation, and test sets
train_embeddings = extract_embeddings(tabnet, X_train_tensor)
valid_embeddings = extract_embeddings(tabnet, X_valid_tensor)
test_embeddings = extract_embeddings(tabnet, X_test_tensor)

# Optional: Save the embeddings for future use
np.save('train_embeddings.npy', train_embeddings)
np.save('valid_embeddings.npy', valid_embeddings)
np.save('test_embeddings.npy', test_embeddings)

# Visualize embeddings or use them in downstream tasks
print("Embeddings generated and saved successfully.")