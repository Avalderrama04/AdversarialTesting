import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from torch import nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_paths = []
        self.labels = []
        
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                if image_name.endswith(".jpg") or image_name.endswith(".png"):
                    self.image_paths.append(os.path.join(class_dir, image_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

resize_transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor()  
])

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
class_id_to_name = {i: class_name for i, class_name in enumerate(class_names)}
class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

class FineTunedDenseNet(nn.Module):
     def __init__(self, pretrained_model, num_classes=4):
        super(FineTunedDenseNet, self).__init__()
        self.features = pretrained_model.features  
        self.classifier = nn.Linear(pretrained_model.classifier.in_features, num_classes)  

     def forward(self, x):
        x = self.features(x) 
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1) 
        x = self.classifier(x)  
        return x

     def train_model(self, model_name, input_size, epochs, learning_rate, train_loader, test_loader, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        best_accuracy = 0
        train_loss_values = []
        test_accuracy_values = []

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=100)

            for i, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                progress_bar.set_postfix(loss=running_loss / (i + 1), accuracy=100 * correct / total)

            test_accuracy = 100 * correct / total
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(self.state_dict(), "best_densenet_model.pth")

            scheduler.step()  
            train_loss_values.append(running_loss / len(train_loader))
            test_accuracy_values.append(test_accuracy)

        self.plot_loss(model_name, input_size, learning_rate, train_loss_values, test_accuracy_values)

     def plot_loss(self, model_name, input_size, learning_rate, train_loss_values, test_accuracy_values):
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_values, label="Training Loss", color='blue')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"training_loss_{model_name}_{input_size}_lr{learning_rate}.png")
        plt.show()
    
        plt.figure(figsize=(10, 5))
        plt.plot(test_accuracy_values, label="Test Accuracy", color='orange')
        plt.xlabel("Steps")
        plt.ylabel("Accuracy (%)")
        plt.title("Test Accuracy Over Time")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"test_accuracy_{model_name}_{input_size}_lr{learning_rate}.png")
        plt.show()

     def save(self, model_file):
        torch.save(self.state_dict(), model_file)

if __name__ == "__main__":
    root_dir = '/Users/arthe/honors/brain/training'
    model_name = 'FineTunedDenseNet'
    batch_size = 32
    learning_rate = 0.0003
    epochs = 10
    input_size = 64

    image_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = CustomDataset(root_dir, transform=image_transforms)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training {model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    pretrained_densenet = models.densenet121(pretrained=True)
    model = FineTunedDenseNet(pretrained_model=pretrained_densenet, num_classes=4)
    model.to(device)

    model.train_model(model_name, input_size, epochs, learning_rate, train_loader, test_loader, device=device)

    model.to('cpu')
    model_path = f"{model_name}_{input_size}_lr{learning_rate}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
