import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split # Thêm random_split để chia tập xác thực
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import seaborn as sns

#1. Load and process CIFAR-10 data
print("Loading and preprocessing CIFAR-10 data...")
transform = transforms.ToTensor() 

data_path = '../data_cifar/' 
cifar10_train_full = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

# Split the training set into training set and validation set
val_split_ratio = 0.2
train_size = int((1 - val_split_ratio) * len(cifar10_train_full))
val_size = len(cifar10_train_full) - train_size
cifar10_train, cifar10_val = random_split(cifar10_train_full, [train_size, val_size])


# Set seeds to ensure reproducible results
torch.manual_seed(80)

# DataLoader
batch_size_train = 100
batch_size_test = 500
batch_size_val = 500 

train_loader = DataLoader(cifar10_train, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(cifar10_val, batch_size=batch_size_val, shuffle=False) # DataLoader cho tập xác thực
test_loader = DataLoader(cifar10_test, batch_size=batch_size_test, shuffle=False)


# 2. MLP
class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size=32*32*3, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 120) 
        self.fc2 = nn.Linear(120, 84)        
        self.fc3 = nn.Linear(84, output_size) 

    def forward(self, X):
        X = F.relu(self.fc1(X)) 
        X = F.relu(self.fc2(X)) 
        X = self.fc3(X)        
        return X

# 3.Model initialization, loss function, and optimization
torch.manual_seed(80)
model = MultilayerPerceptron()
print("\nMLP model architecture:")
print(model)


total_params = 0
for param in model.parameters():
    total_params += param.numel()
print(f"Total number of model parameters: {total_params}")

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

# 4. Model training
print("\nStart training MLP model...")
epochs = 10
train_losses, val_losses = [], [] 
train_accuracies, val_accuracies = [], [] 

start_time = time.time()

for i in range(epochs):
    model.train() 
    running_train_loss = 0.0
    correct_train_predictions = 0
    total_train_samples = 0

    for X_train, y_train in train_loader:
        X_train_flat = X_train.view(X_train.shape[0], -1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X_train_flat)
        loss = criterion(y_pred, y_train)

        
        loss.backward()
        optimizer.step()

        
        predicted = torch.max(y_pred, 1)[1] 
        correct_train_predictions += (predicted == y_train).sum().item()
        total_train_samples += y_train.size(0)
        running_train_loss += loss.item() * X_train.size(0)

    epoch_train_loss = running_train_loss / total_train_samples
    epoch_train_accuracy = correct_train_predictions / total_train_samples
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    
    model.eval() 
    running_val_loss = 0.0
    correct_val_predictions = 0
    total_val_samples = 0
    with torch.no_grad(): 
        for X_val, y_val in val_loader:
            X_val_flat = X_val.view(X_val.shape[0], -1)
            y_val_pred = model(X_val_flat)
            
            val_loss = criterion(y_val_pred, y_val)
            
            predicted_val = torch.max(y_val_pred, 1)[1]
            correct_val_predictions += (predicted_val == y_val).sum().item()
            total_val_samples += y_val.size(0)
            running_val_loss += val_loss.item() * X_val.size(0)

    epoch_val_loss = running_val_loss / total_val_samples
    epoch_val_accuracy = correct_val_predictions / total_val_samples
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    print(f'Epoch: {i+1}/{epochs} | '
          f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.4f} | '
          f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.4f}')


total_time = time.time() - start_time
print(f'\nTraining time: {total_time/60:.2f} phút')

# 5. Plotting Loss and Accuracy Charts
print("\nPlotting training history graph...")
plt.figure(figsize=(10, 5))

epochs = range(1, len(train_losses) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


ax1.plot(epochs, train_losses, label='Training Loss')
ax1.plot(epochs, val_losses, label='Validation Loss')
ax1.set_title('MLP - Loss Curves')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, train_accuracies, label='Training Accuracy')
ax2.plot(epochs, val_accuracies, label='Validation Accuracy')
ax2.set_title('MLP - Accuracy Curves')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()

fig.suptitle('Learning Curves for MLP', fontsize=14)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 6.Evaluation on final test set
print("\nevaluating the model on the final test set...")
model.eval() 
correct_test_predictions = 0
total_test_samples = 0
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test_flat = X_test.view(X_test.shape[0], -1)
        y_test_pred = model(X_test_flat)
        predicted_test = torch.max(y_test_pred, 1)[1]
        correct_test_predictions += (predicted_test == y_test).sum().item()
        total_test_samples += y_test.size(0)

final_test_accuracy = correct_test_predictions / total_test_samples
print(f'Final accuracy on the test set of MLP: {final_test_accuracy:.4f}')

# 7.Confusion Matrix
print("\nCreating a confusion matrix...")
all_preds = []
all_labels = []
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test_flat = X_test.view(X_test.shape[0], -1)
        y_test_pred = model(X_test_flat)
        predicted = torch.max(y_test_pred, 1)[1]
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("\nconfusion matrix:")
print(cm)

plt.figure(figsize=(10, 8))
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn - MLP trên CIFAR-10')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


print("\nVisualizing predictions for some sample images from the test set...")
classes = cifar10_train_full.classes 
test_display_loader = DataLoader(cifar10_test, batch_size=64, shuffle=True) 

images_display, labels_display = next(iter(test_display_loader))

with torch.no_grad():
    images_display_flat = images_display.view(images_display.shape[0], -1)
    preds_display = model(images_display_flat)
    predicted_classes = torch.max(preds_display, 1)[1] 

fig = plt.figure(figsize=(12, 12))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.5, wspace=0.05)
 
for i in range(min(64, images_display.shape[0])): 
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(images_display[i].permute(1, 2, 0).cpu().numpy())
    
    true_label = classes[labels_display[i].item()]
    predicted_label = classes[predicted_classes[i].item()]
    
    color = "blue" if labels_display[i] == predicted_classes[i] else "red"
    title_text = f"True: {true_label}\nPred: {predicted_label}"
    plt.title(title_text, color=color, fontsize=10)
plt.show()
print("\nDone.")
