import os
import torch
import time  
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt  
from PIL import Image

from train import class_names, class_id_to_name, class_to_idx, CustomDataset
from train import FineTunedDenseNet  

model_file = "/Users/arthe/honors/FineTunedDenseNet_64_lr0.0003.pt"

script_dir = os.path.dirname(os.path.abspath(__file__))
adversarial_save_dir = os.path.join(script_dir, "adversarial_images1")
os.makedirs(adversarial_save_dir, exist_ok=True)

resize_transform = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)

full_transform = transforms.Compose([
    resize_transform,
    transforms.ToTensor()  
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_densenet = models.densenet121(pretrained=True)

model = FineTunedDenseNet(pretrained_model=pretrained_densenet, num_classes=len(class_names))
model.load_state_dict(torch.load(model_file, map_location=device))
model.to(device)
model.eval()

def cw_l2_attack(model, images, labels, target_labels=None, targeted=False, c=2.5, kappa=0, max_iter=1000, learning_rate=0.01):
    """
    CW-L2 Attack to generate adversarial examples on full-resolution images.
    Args:
        model (nn.Module): Trained model to attack.
        images (torch.Tensor): Batch of full-resolution images to attack.
        labels (torch.Tensor): True labels for the images.
        target_labels (torch.Tensor, optional): Target labels for the attack. Required if targeted=True.
        targeted (bool): Whether the attack is targeted or untargeted.
        c (float): Regularization parameter to control attack strength.
        kappa (float): Confidence parameter for the attack.
        max_iter (int): Maximum number of optimization iterations.
        learning_rate (float): Learning rate for the optimizer.
    Returns:
        torch.Tensor: Adversarial examples in full resolution.
    """

    images = images.to(device)
    labels = labels.to(device)

    if targeted:
        if target_labels is None:
            raise ValueError("Target labels must be provided for targeted attacks.")
        target_labels = target_labels.to(device)

    def f(x):
        resized_x = resize_transform(x)
        outputs = model(resized_x)

        if targeted:
            one_hot_labels = torch.eye(outputs.shape[1], device=device)[target_labels]
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            return torch.clamp(i - j, min=-kappa)
        else:
            one_hot_labels = torch.eye(outputs.shape[1], device=device)[labels]
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate)
    prev = 1e10

    for step in range(max_iter):
        a = 0.5 * (torch.tanh(w) + 1)

        loss1 = F.mse_loss(a, images, reduction='sum')
        loss2 = torch.sum(c * f(a))
        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if step % (max_iter // 10) == 0:
            if cost > prev:
                print("Attack Stopped due to CONVERGENCE....")
                return a
            prev = cost

        if (step + 1) % 100 == 0 or step == 0:
            print(f"- Learning Progress: {(step + 1) / max_iter * 100:.2f}%", end="\r")

    attack_images = 0.5 * (torch.tanh(w) + 1)
    return attack_images

def save_image(image, true_label, predicted_label, original_size, index, prefix="adversarial_example"):
   
    image = transforms.ToPILImage()(image.cpu().detach())

    image_resized = image.resize(original_size, Image.LANCZOS)

    os.makedirs(adversarial_save_dir, exist_ok=True)

    file_name = os.path.join(
        adversarial_save_dir,
        f"{prefix}_{index}_true_{true_label}_pred_{predicted_label}.png"
    )

    image_resized.save(file_name)

def display_images(original_image, adversarial_image, true_label, predicted_label):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_image.permute(1, 2, 0).cpu().detach().numpy())  
    axes[0].set_title(f"Original: {true_label}")
    axes[0].axis('off')

    axes[1].imshow(adversarial_image.permute(1, 2, 0).cpu().detach().numpy())  
    axes[1].set_title(f"Adversarial: {predicted_label}")
    axes[1].axis('off')

    plt.show()

if __name__ == "__main__":
    import argparse

    targeted = False
    target_class = "glioma"
    c = 20
    kappa = 0
    learning_rate = 0.01
    max_iter = 1000

    data_folder = "/Users/arthe/honors/brain/training" 
    full_dataset = CustomDataset(data_folder, transform=full_transform)
    train_size = int(0.8 * len(full_dataset))  
    test_size = len(full_dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    batch_size = 1
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    attack_accuracy = 0
    correct = 0
    total = 0 

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        original_size = images.shape[2], images.shape[3] 

        start_time = time.time() 

        if targeted:
            if target_class is None:
                raise ValueError("Target class must be specified for targeted attacks.")
            try:
                target_class_idx = class_to_idx[target_class]
            except KeyError:
                raise ValueError(f"Target class '{target_class}' not found in class_to_idx.")
            target_labels = torch.full_like(labels, target_class_idx)
            adversarial_images = cw_l2_attack(
                model, images, labels, target_labels=target_labels, targeted=True, c=c, 
                kappa=kappa, learning_rate=learning_rate, max_iter=max_iter
            )
        else:
            adversarial_images = cw_l2_attack(
                model, images, labels, targeted=False, c=c, 
                kappa=kappa, learning_rate=learning_rate, max_iter=max_iter
            )

        end_time = time.time()  
        attack_time = end_time - start_time
        print(f"Time taken for attack: {attack_time:.2f} seconds")

        adversarial_images_resized = resize_transform(adversarial_images)

        outputs = model(adversarial_images_resized)
        _, predicted = torch.max(outputs, 1)

        if targeted:
            batch_correct = (predicted == target_labels).sum().item()
        else:
            batch_correct = (predicted != labels).sum().item()

        batch_total = labels.size(0)
        attack_accuracy += 100 * batch_correct / batch_total
        correct += batch_correct
        total += batch_total

        print(f"Attack Accuracy on the batch: {100 * batch_correct / batch_total:.2f}%")

        for i in range(len(adversarial_images)):
            true_label = class_names[labels[i].item()]
            predicted_label = class_names[predicted[i].item()]

            save_image(adversarial_images[i], true_label, predicted_label, original_size, i)

            display_images(images[i], adversarial_images[i], true_label, predicted_label)

    overall_attack_accuracy = attack_accuracy / len(test_loader)
    print(f"\nOverall Attack Accuracy: {overall_attack_accuracy:.2f}%")
