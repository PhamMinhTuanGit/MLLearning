import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# ==== Đọc dữ liệu ====
DATA_ROOT = "/home/parallels/Documents/code/MLlearning/Housing_processed.csv"
data = pd.read_csv(DATA_ROOT)

labels = data["price"]
inputs = data.drop("price", axis=1)
print("Tổng shape:", inputs.shape)
from sklearn.preprocessing import StandardScaler

# Chuẩn hóa input
scaler = StandardScaler()
inputs = pd.DataFrame(scaler.fit_transform(inputs), columns=inputs.columns)

# Chuẩn hóa label
label_scaler = StandardScaler()
labels = label_scaler.fit_transform(labels.values.reshape(-1, 1))
# ==== Tách train/test ====
inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.2)
print("Train shape:", inputs_train.shape)

# ==== Convert sang Tensor đúng cách ====
inputs_train = torch.tensor(inputs_train.values, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.float32).view(-1, 1)
inputs_test = torch.tensor(inputs_test.values, dtype=torch.float32)
labels_test = torch.tensor(labels_test, dtype=torch.float32).view(-1, 1)

# ==== Tạo DataLoader ====
train_dataset = TensorDataset(inputs_train, labels_train)
test_dataset = TensorDataset(inputs_test, labels_test)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==== models ====
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
# ==== train ====
model = LinearRegression(9, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
for epoch in range(200):
    model.train()
    total_loss = 0

    for batch_X, batch_y in trainloader:
        preds = model(batch_X)
        loss = criterion(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)

    avg_loss = total_loss / len(trainloader.dataset)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")
# ==== test ====
model.eval()
with torch.no_grad():
    test_loss = 0
    for batch_X, batch_y in testloader:
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        test_loss += loss.item() * batch_X.size(0)

    avg_test_loss = test_loss / len(testloader.dataset)
    print(f"\n🎯 Test Loss (MSE): {avg_test_loss:.4f}")

import matplotlib.pyplot as plt

# Lấy toàn bộ tập test
X_test = inputs_test
y_test = labels_test

# Dự đoán
model.eval()
with torch.no_grad():
    predictions = model(X_test)

# Chuyển về numpy để vẽ
y_test_np = y_test.numpy()
preds_np = predictions.numpy()

# Vẽ
plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, preds_np, alpha=0.6, color='b', label='Dự đoán')
plt.plot([y_test_np.min(), y_test_np.max()],
         [y_test_np.min(), y_test_np.max()],
         color='r', linestyle='--', label='Đường lý tưởng (y = x)')

plt.xlabel("Giá trị thực tế (Actual Price)")
plt.ylabel("Giá trị dự đoán (Predicted Price)")
plt.title("Biểu đồ: Dự đoán vs Thực tế")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("linear_regression_result.png")

