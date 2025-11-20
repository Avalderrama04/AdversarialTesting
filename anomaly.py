import torch
from train import FineTunedDenseNet
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models
import os
from sklearn.model_selection import train_test_split

original_data_dir = '/Users/arthe/honors/train'
adversarial_data_dir = '/Users/arthe/honors/adversarial_images'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

original_dataset = datasets.ImageFolder(
    root=original_data_dir,
    transform=transform
)

adversarial_dataset = datasets.ImageFolder(
    root=adversarial_data_dir,
    transform=transform
)

class RelabeledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, new_label):
        self.dataset = dataset
        self.new_label = new_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, self.new_label

original_binary = RelabeledDataset(original_dataset, 0) 
adversarial_binary = RelabeledDataset(adversarial_dataset, 1)  
combined_dataset = ConcatDataset([original_binary, adversarial_binary])

all_data = [(img, label) for img, label in combined_dataset]
labels = [label for _, label in all_data]
train_idx, val_idx = train_test_split(
    list(range(len(all_data))), test_size=0.2, stratify=labels, random_state=42
)
train_dataset = Subset(combined_dataset, train_idx)
val_dataset = Subset(combined_dataset, val_idx)

class AnomalyDenseNet(torch.nn.Module):
    def __init__(self, base_model: FineTunedDenseNet):
        super(AnomalyDenseNet, self).__init__()
        self.features = base_model.features  

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 1)  
        )

    def forward(self, x):
        x = self.features(x)
        out = self.classifier(x)
        return out.squeeze() 


    def train_model(self, model_name, input_size, epochs, learning_rate, train_loader, val_loader, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            self.train()
            running_loss, correct_train, total_train = 0.0, 0, 0
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

            for inputs, labels in train_loader_tqdm:
                inputs, labels = inputs.to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = self(inputs).view(-1)
                labels = labels.view(-1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_train += (preds == labels).sum().item()
                total_train += batch_size

                train_loader_tqdm.set_postfix({
                    'loss': f"{running_loss/total_train:.4f}",
                    'acc': f"{correct_train/total_train:.4f}"
                })

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct_train / total_train

            self.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

            with torch.no_grad():
                for inputs, labels in val_loader_tqdm:
                    inputs, labels = inputs.to(device), labels.float().to(device)
                    outputs = self(inputs).view(-1)
                    loss = criterion(outputs, labels)
                    batch_size = inputs.size(0)
                    val_loss += loss.item() * batch_size
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct_val += (preds == labels).sum().item()
                    total_val += batch_size

                    val_loader_tqdm.set_postfix({
                        'val_loss': f"{val_loss/total_val:.4f}",
                        'val_acc': f"{correct_val/total_val:.4f}"
                    })

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc = correct_val / total_val

            print(f"\nEpoch {epoch+1}/{epochs} Complete")
            print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
            print(f"Val   Loss: {val_epoch_loss:.4f}, Val   Acc: {val_epoch_acc:.4f}")
            print("--------------------------------------------------")

if __name__ == "__main__":
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_densenet = models.densenet121(pretrained=True)  
    base_model = FineTunedDenseNet(pretrained_model=original_densenet, num_classes=4)  
    base_model.load_state_dict(torch.load("/Users/arthe/honors/best_densenet_model.pth", map_location=device))
    base_model = base_model.to(device)

    model = AnomalyDenseNet(base_model).to(device)

    model.train_model("anomaly_model", batch_size, 2, 0.001, train_loader, val_loader, device=device)

    model.to('cpu')
    torch.save(model.state_dict(), "anomaly_detection_model.pth")

    print("\n Checking prediction scores on normal (clean) images...")
    model.eval()
    for i in range(5):
        img, _ = original_binary[i]
        input_tensor = img.unsqueeze(0)
        with torch.no_grad():
            raw_output = model(input_tensor)
            score = torch.sigmoid(raw_output).item()
            print(f"[Clean] Image {i+1} Score: {score:.4f} (Expected: ~0)")

    print("\nChecking prediction scores on adversarial images...")
    for i in range(5):
        img, _ = adversarial_binary[i]
        input_tensor = img.unsqueeze(0)
        with torch.no_grad():
            raw_output = model(input_tensor)
            score = torch.sigmoid(raw_output).item()
            print(f"[Adversarial] Image {i+1} Score: {score:.4f} (Expected: ~1)")