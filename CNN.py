import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
class CNN(nn.Module):
    def __init__(self, dropout_rate=0.3): # Added dropout_rate parameter
        super(CNN, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Added BatchNorm
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # Added BatchNorm
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # Added BatchNorm
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128 x 4 x 4
 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout_rate) # Added Dropout
        self.fc2 = nn.Linear(512, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) # Apply BatchNorm
        x = self.relu1(x)
        x = self.pool1(x)
 
        x = self.conv2(x)
        x = self.bn2(x) # Apply BatchNorm
        x = self.relu2(x)
        x = self.pool2(x)
 
        x = self.conv3(x)
        x = self.bn3(x) # Apply BatchNorm
        x = self.relu3(x)
        x = self.pool3(x)
 
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout_fc(x) # Apply Dropout
        x = self.fc2(x)
        return x
 
def train_model(model, trainloader, criterion, optimizer, device, num_epochs=10, valloader=None):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
 
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
 
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
 
        epoch_train_loss = running_loss / len(trainloader)
        epoch_train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
 
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%", end="")
 
        if valloader:
            model.eval() # Set model to evaluation mode
            val_loss_epoch = 0.0
            correct_val_epoch = 0
            total_val_epoch = 0
            with torch.no_grad():
                for data_val in valloader:
                    images_val, labels_val = data_val
                    images_val, labels_val = images_val.to(device), labels_val.to(device)
                    outputs_val = model(images_val)
                    loss_v = criterion(outputs_val, labels_val)
                    val_loss_epoch += loss_v.item()
                    _, predicted_val = torch.max(outputs_val.data, 1)
                    total_val_epoch += labels_val.size(0)
                    correct_val_epoch += (predicted_val == labels_val).sum().item()
 
            epoch_val_loss = val_loss_epoch / len(valloader)
            epoch_val_acc = 100 * correct_val_epoch / total_val_epoch
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)
            print(f", Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        else:
            print()
 
    print('Finished Training')
    return train_losses, train_accuracies, val_losses, val_accuracies
 
# 3.2. Hàm kiểm tra mô hình
def test_model(model, testloader, device):
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
 
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {total} test images: {accuracy:.2f} %')
    return accuracy, all_predicted, all_labels
 
# 4. Hàm vẽ đồ thị quá trình học
def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, model_name="Model"):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
 
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    if val_losses:
        plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.legend()
 
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    if val_accuracies:
        plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Accuracy Curves')
    plt.legend()
 
    plt.suptitle(f'Learning Curves for {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.show()
 
# 5. Hàm vẽ ma trận nhầm lẫn
def plot_confusion_matrix(true_labels, predicted_labels, class_names, model_name="Model"):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()
 
# --- BẮT ĐẦU KHỐI MÃ CHÍNH ---
if __name__ == '__main__':
    # --- Cài đặt chung ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    num_epochs_global = 25 # Increased epochs as regularization might allow for longer training
    learning_rate_global = 0.001
    batch_size_global = 64
    num_workers_global = 2 # Set to 0 if you have issues with multiprocessing on Windows
    weight_decay_global = 1e-4 # Added weight decay for regularization
    dropout_rate_global = 0.3  # Defined a global dropout rate
 
    # --- Chuẩn bị dữ liệu (CIFAR-10) ---
    # Added Data Augmentation for training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
 
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
 
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_global,
                                              shuffle=True, num_workers=num_workers_global)
 
    # Using testset for validation during training and also for final testing
    # For a more rigorous approach, a separate validation set from the training data could be created.
    validationset = torchvision.datasets.CIFAR10(root='./data', train=False, # Using test data as validation
                                           download=True, transform=transform_test)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size_global,
                                             shuffle=False, num_workers=num_workers_global)
 
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_global,
                                             shuffle=False, num_workers=num_workers_global)
 
 
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # --- Khởi tạo và Huấn luyện CNN Model ---
    print("\n--- Initializing CNN Model ---")
    cnn_model = CNN(dropout_rate=dropout_rate_global).to(device)
    print(cnn_model)
    criterion_cnn = nn.CrossEntropyLoss()
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=learning_rate_global, weight_decay=weight_decay_global) # Added weight_decay
 
    print("\n--- Training CNN Model ---")
    cnn_train_losses, cnn_train_accuracies, cnn_val_losses, cnn_val_accuracies = train_model(
        cnn_model, trainloader, criterion_cnn, optimizer_cnn, device,
        num_epochs=num_epochs_global, valloader=valloader # Pass valloader
    )

    # --- Kiểm tra mô hình trên tập test ---
    # Note: The validation set used during training is the same as the test set here.
    # In a typical ML workflow, the test set is held out and used only once for final evaluation.
    print("\n--- Testing CNN Model ---")
    cnn_test_accuracy, cnn_predicted, cnn_labels = test_model(cnn_model, testloader, device)

    print("\nPlotting Learning Curves for CNN...")
    plot_learning_curves(cnn_train_losses, cnn_val_losses, cnn_train_accuracies, cnn_val_accuracies, "CNN")

    #Vẽ ma trận nhầm lẫnlẫn
    print("\nPlotting Confusion Matrix for CNN...")
    plot_confusion_matrix(cnn_labels, cnn_predicted, classes, "CNN")

    # --- So sánh và Thảo luận kết quả ---
    print("\n--- Results Summary ---")
    print(f"CNN Final Test Accuracy (after {num_epochs_global} epochs): {cnn_test_accuracy:.2f}%")
 