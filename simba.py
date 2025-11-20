import torch
import utils
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import densenet121  
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import time
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimBA:
    
    def __init__(self, model, dataset, image_size):
        self.model = model.to(device) 
        self.dataset = dataset
        self.image_size = image_size
        self.model.eval()
    
    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z
        
    def normalize(self, x):
        return utils.apply_normalization(x.to(device), self.dataset)


    def get_probs(self, x, y):
        output = self.model(self.normalize(x)).cpu()
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        return torch.diag(probs)
    
    def get_preds(self, x):
        output = self.model(self.normalize(x)).cpu()
        _, preds = output.data.max(1)
        return preds

    def simba_single(self, x, y, num_iters=10000, epsilon=2.5, targeted=False):
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        last_prob = self.get_probs(x, y)
        for i in range(num_iters):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if targeted != (right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if i % 10 == 0:
                print(last_prob)
        return x.squeeze()

   
    def simba_batch(self, images_batch, labels_batch, max_iters, freq_dims, stride, epsilon, linf_bound=0.0,
                    order='rand', targeted=False, pixel_attack=False, log_every=1):
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)
        assert self.image_size == image_size
        if order == 'rand':
            indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
        elif order == 'diag':
            indices = utils.diagonal_order(image_size, 3)[:max_iters]
        elif order == 'strided':
            indices = utils.block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
        else:
            indices = utils.block_order(image_size, 3)[:max_iters]
        if order == 'rand':
            expand_dims = freq_dims
        else:
            expand_dims = image_size
        n_dims = 3 * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims)
        probs = torch.zeros(batch_size, max_iters)
        succs = torch.zeros(batch_size, max_iters)
        queries = torch.zeros(batch_size, max_iters)
        l2_norms = torch.zeros(batch_size, max_iters)
        linf_norms = torch.zeros(batch_size, max_iters)
        prev_probs = self.get_probs(images_batch, labels_batch)
        preds = self.get_preds(images_batch)
        if pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: utils.block_idct(z, block_size=image_size, linf_bound=linf_bound)
        remaining_indices = torch.arange(0, batch_size).long()
        for k in range(max_iters):
            dim = indices[k]
            expanded = (images_batch[remaining_indices] + trans(self.expand_vector(x[remaining_indices], expand_dims))).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims))
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            preds_next = self.get_preds(expanded)
            preds[remaining_indices] = preds_next
            if targeted:
                remaining = preds.ne(labels_batch)
            else:
                remaining = preds.eq(labels_batch)
            if remaining.sum() == 0:
                adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break
            remaining_indices = torch.arange(0, batch_size)[remaining].long()
            if k > 0:
                succs[:, k] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            diff[:, dim] = epsilon
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(left_vec, expand_dims))).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size)
            queries_k[remaining_indices] += 1
            if targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                improved = left_probs.lt(prev_probs[remaining_indices])
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(right_vec, expand_dims))).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            if targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
            probs_k = prev_probs.clone()
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            if (k + 1) % log_every == 0 or k == max_iters - 1:
                print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                        k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
        expanded = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
        preds = self.get_preds(expanded)
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        succs[:, max_iters-1] = ~remaining
        return expanded, probs, succs, queries, l2_norms, linf_norms
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
])


dataset_path = '/Users/arthe/honors/brain/training' 
dataset = ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = densenet121(pretrained=True)
model.classifier = torch.nn.Linear(in_features=1024, out_features=4)


model.load_state_dict(torch.load('/Users/arthe/honors/FineTunedDenseNet_64_lr0.0003.pt', map_location=device))
model.eval()

simba = SimBA(model=model, dataset='imagenet', image_size=224)#was 224

def simba_single_attack(simba, dataloader):
    images_batch, labels_batch = next(iter(dataloader))
    image = images_batch[0:1].to(device)  
    label = labels_batch[0:1].to(device)

    class_name = dataloader.dataset.classes[label.item()]

    start_time = time.time()
    adv_image = simba.simba_single(image, label, num_iters=100, epsilon=0.2, targeted=False)
    elapsed_time = time.time() - start_time

    adv_pred = simba.get_preds(adv_image.unsqueeze(0))
    adv_class_name = dataloader.dataset.classes[adv_pred.item()]

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
    plt.title(f"Original: {class_name} ({label.item()})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(adv_image.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Adversarial: {adv_class_name} ({adv_pred.item()})")
    plt.axis('off')

    plt.suptitle(f"SimBA Single Attack | Time: {elapsed_time:.2f}s")
    plt.show()



def simba_full_dataset_attack(simba, dataloader, max_iters=200, epsilon=0.2, freq_dims=21, stride=1, targeted=False):
    total_images = 0
    total_fooled = 0
    total_time = 0.0

    for images_batch, labels_batch in dataloader:
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)

        start_time = time.time()
        adv_images, probs, succs, queries, l2_norms, linf_norms = simba.simba_batch(
            images_batch=images_batch, 
            labels_batch=labels_batch, 
            max_iters=max_iters,  
            freq_dims=freq_dims, 
            stride=stride, 
            epsilon=epsilon,  
            linf_bound=0.0, 
            order='rand', 
            targeted=targeted, 
            pixel_attack=False, 
            log_every=50
        )
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        adv_preds = simba.get_preds(adv_images)
        fooled = (adv_preds != labels_batch).sum().item()
        total_fooled += fooled
        total_images += len(labels_batch)

        print(f"Batch: {total_images}/{len(dataloader.dataset)} - Fooled: {fooled}/{len(labels_batch)} - Time: {elapsed_time:.2f}s")

    success_rate = (total_fooled / total_images) * 100
    avg_time = total_time / len(dataloader)

    print(f"\nSimBA Attack Completed")
    print(f"Total Images: {total_images}")
    print(f"Total Fooled: {total_fooled}")
    print(f"Attack Success Rate: {success_rate:.2f}%")
    print(f"Average Time per Batch: {avg_time:.2f}s")

    return success_rate



def evaluate_adversarial_attacks(model, dataloader, targeted=False, max_iters=200, freq_dims=21, stride=1, epsilon=2.5):
    
    model.eval() 
    attack_accuracy = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        start_time = time.time()

        if targeted:
            if target_class is None:
                raise ValueError("Target class must be specified for targeted attacks.")
            try:
                target_class_idx = dataloader.dataset.class_to_idx[target_class]
            except KeyError:
                raise ValueError(f"Target class '{target_class}' not found in class_to_idx.")
            target_labels = torch.full_like(labels, target_class_idx)
            adv_images, *_ = simba.simba_batch(
                images_batch=images, 
                labels_batch=labels, 
                max_iters=max_iters,
                freq_dims=freq_dims,
                stride=stride,
                epsilon=epsilon,
                targeted=True
            )
        else:
            adv_images, *_ = simba.simba_batch(
                images_batch=images, 
                labels_batch=labels, 
                max_iters=max_iters,
                freq_dims=freq_dims,
                stride=stride,
                epsilon=epsilon,
                targeted=False
            )

        end_time = time.time()
        attack_time = end_time - start_time
        print(f"Time taken for attack: {attack_time:.2f} seconds")

        outputs = model(adv_images)
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

    overall_attack_accuracy = attack_accuracy / len(dataloader)
    print(f"\nOverall Attack Accuracy: {overall_attack_accuracy:.2f}%")

    return overall_attack_accuracy

#simba_single_attack(simba, dataloader)

#success_rate = simba_full_dataset_attack(simba, dataloader, max_iters=200, epsilon=0.2)
overall_accuracy = evaluate_adversarial_attacks(model, dataloader, targeted=False, max_iters=200)
