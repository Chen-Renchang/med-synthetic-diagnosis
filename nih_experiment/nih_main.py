# -*- coding: utf-8 -*-
"""
Эксперимент на NIH Chest X-ray: Hernia (редкий) против Infiltration (норма).
Условный DDPM с коморбидными условиями (Infiltration, Atelectasis, Effusion).
Визуализация сравнения реальных и сгенерированных изображений.
Исправлено отображение SMOTE в матрицах ошибок.
Весь вывод на русском языке.
"""

import os, glob, math, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.colors

try:
    from skimage import exposure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("scikit-image не установлен, CLAHE не будет применён.")

warnings.filterwarnings('ignore')

# Параметры
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

DATA_BASE = '/kaggle/input/datasets/organizations/nih-chest-xrays/data'
CSV_PATH = os.path.join(DATA_BASE, 'Data_Entry_2017.csv')
MAJOR_CLASS = 'Infiltration'
MINOR_CLASS = 'Hernia'
SELECTED_DISEASES = ['Infiltration', 'Atelectasis', 'Effusion']  # условия

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_CLS = 20
EPOCHS_GAN = 100
EPOCHS_DIFFUSION = 100          # число эпох обучения DDPM
NUM_SYNTHETIC = 200
MAX_SAMPLES = 800
DIFFUSION_TIMESTEPS = 200

# Построение карты изображений
def build_image_path_map(base_dir):
    path_map = {}
    subdirs = glob.glob(os.path.join(base_dir, 'images_*'))
    if not subdirs:
        alt_dir = os.path.join(base_dir, 'images')
        if os.path.isdir(alt_dir):
            subdirs = [alt_dir]
        else:
            raise FileNotFoundError(f"Нет директорий с изображениями в {base_dir}")
    for subdir in subdirs:
        img_dir = os.path.join(subdir, 'images')
        if not os.path.isdir(img_dir):
            img_dir = subdir
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path_map[fname] = os.path.join(img_dir, fname)
    print(f"Найдено {len(path_map)} изображений.")
    return path_map

def load_data_multilabel(csv_path, path_map, major_class, minor_class, selected_diseases, max_samples=None):
    df = pd.read_csv(csv_path)
    df = df[df['Image Index'].isin(path_map.keys())].copy()
    print(f"После фильтрации по карте путей: {len(df)} сэмплов")

    df_minor = df[df['Finding Labels'].str.contains(minor_class)].copy()
    df_major = df[df['Finding Labels'] == major_class].copy()
    if max_samples and len(df_major) > max_samples:
        df_major = df_major.sample(n=max_samples, random_state=seed)

    df_combined = pd.concat([df_major, df_minor], ignore_index=True)
    df_combined['label'] = [0] * len(df_major) + [1] * len(df_minor)
    df_combined['image_path'] = df_combined['Image Index'].apply(lambda x: path_map[x])
    df_combined['Patient Age'] = pd.to_numeric(df_combined['Patient Age'], errors='coerce').fillna(
        df_combined['Patient Age'].median())
    df_combined['gender_code'] = df_combined['Patient Gender'].map({'M': 0, 'F': 1}).fillna(0).astype(int)

    for disease in selected_diseases:
        df_combined[disease] = df_combined['Finding Labels'].apply(
            lambda x: 1 if disease in x.split('|') else 0)
    return df_combined

print("Построение карты путей...")
image_path_map = build_image_path_map(DATA_BASE)
df = load_data_multilabel(CSV_PATH, image_path_map, MAJOR_CLASS, MINOR_CLASS,
                          SELECTED_DISEASES, max_samples=MAX_SAMPLES)
print(f"Всего сэмплов: {len(df)}")
print(f"Класс '{MAJOR_CLASS}': {(df['label']==0).sum()}, класс '{MINOR_CLASS}': {(df['label']==1).sum()}")
print(f"Условия для модели E/F: {SELECTED_DISEASES}")

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=seed)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=seed)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Аугментация
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None, selected_diseases=None):
        self.df = dataframe
        self.transform = transform
        self.selected_diseases = selected_diseases if selected_diseases is not None else []
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        label = row['label']
        comorbidities = torch.tensor([row[d] for d in self.selected_diseases], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': torch.tensor(label, dtype=torch.long), 'comorbidities': comorbidities}

train_dataset = ChestXrayDataset(train_df, transform=train_transform, selected_diseases=SELECTED_DISEASES)
val_dataset = ChestXrayDataset(val_df, transform=eval_transform, selected_diseases=SELECTED_DISEASES)
test_dataset = ChestXrayDataset(test_df, transform=eval_transform, selected_diseases=SELECTED_DISEASES)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Модели классификаторов
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        final_size = IMG_SIZE // 8
        self.classifier = nn.Linear(128 * final_size * final_size, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class PretrainedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))
    def forward(self, x):
        return self.backbone(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

# WGAN-GP
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()
        )
    def forward(self, noise):
        z = self.fc(noise).view(-1, 256, 8, 8)
        img = self.deconv(z)
        img = nn.functional.interpolate(img, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)
    def forward(self, img):
        img = nn.functional.interpolate(img, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        feat = self.conv(img)
        feat = self.global_pool(feat).view(feat.size(0), -1)
        return self.fc(feat)

def compute_gradient_penalty(D, real_imgs, fake_imgs):
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolated = D(interpolated)
    grad = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones_like(d_interpolated),
                               create_graph=True, retain_graph=True)[0]
    grad = grad.view(batch_size, -1)
    grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

def train_wgan_gp(generator, discriminator, dataloader, epochs, lambda_gp=10):
    print("Начало обучения WGAN-GP...")
    latent_dim = 100
    optG = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optD = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_iters = 3
    for epoch in range(epochs):
        print(f"WGAN-GP эпоха {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(dataloader):
            real_imgs = batch['image'].to(device)
            bs = real_imgs.size(0)
            for _ in range(d_iters):
                optD.zero_grad()
                noise = torch.randn(bs, latent_dim).to(device)
                fake_imgs = generator(noise)
                d_real = discriminator(real_imgs)
                d_fake = discriminator(fake_imgs.detach())
                gp = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)
                lossD = d_fake.mean() - d_real.mean() + lambda_gp * gp
                lossD.backward()
                optD.step()
            optG.zero_grad()
            fake_imgs = generator(noise)
            d_fake = discriminator(fake_imgs)
            lossG = -d_fake.mean()
            lossG.backward()
            optG.step()
            if (batch_idx+1) % 50 == 0:
                print(f"  batch {batch_idx+1}, LossD: {lossD.item():.4f}, LossG: {lossG.item():.4f}")
        if (epoch+1) % 50 == 0:
            print(f"Эпоха {epoch+1}/{epochs} | LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")
    return generator

# Условный DDPM с CosineAnnealingLR
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class SimpleUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, cond_dim=3, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.ReLU())
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, time_dim), nn.ReLU())
        self.to_feat_channels = nn.ModuleDict({
            '64': nn.Linear(time_dim, 64), '128': nn.Linear(time_dim, 128), '256': nn.Linear(time_dim, 256)})
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2)
        self.mid = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256+256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128+128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64+64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(cond)
        combined = t_emb + c_emb
        def add_emb(feature, name):
            proj = self.to_feat_channels[name](combined)
            proj = proj[:, :, None, None].expand(-1, -1, feature.shape[2], feature.shape[3])
            return feature + proj
        e1 = self.enc1(x); e1 = add_emb(e1, '64')
        p1 = self.pool1(e1)
        e2 = self.enc2(p1); e2 = add_emb(e2, '128')
        p2 = self.pool2(e2)
        e3 = self.enc3(p2); e3 = add_emb(e3, '256')
        p3 = self.pool3(e3)
        m = self.mid(p3)
        d3 = self.up3(m); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.out_conv(d1)

class DDPMGenerator:
    def __init__(self, cond_dim=3, image_size=128, device='cuda', timesteps=DIFFUSION_TIMESTEPS):
        self.device = device
        self.image_size = image_size
        self.timesteps = timesteps
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.model = SimpleUnet(in_channels=3, out_channels=3, cond_dim=cond_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = None

    def train_step(self, real_imgs, cond):
        bs = real_imgs.shape[0]
        t = torch.randint(0, self.timesteps, (bs,), device=self.device).long()
        noise = torch.randn_like(real_imgs)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        noisy_imgs = torch.sqrt(alpha_bar) * real_imgs + torch.sqrt(1 - alpha_bar) * noise
        pred_noise = self.model(noisy_imgs, t, cond)
        return nn.MSELoss()(pred_noise, noise)

    def train(self, dataloader, epochs):
        self.model.train()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                imgs = batch['image'].to(self.device)
                cond = batch['comorbidities'].to(self.device)
                self.optimizer.zero_grad()
                loss = self.train_step(imgs, cond)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
            print(f"Диффузия эпоха {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        return self

    @torch.no_grad()
    def generate(self, cond, num_images=1, noise_std=0.05):
        self.model.eval()
        if noise_std > 0:
            cond = cond + torch.randn_like(cond) * noise_std
        x = torch.randn(num_images, 3, self.image_size, self.image_size).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((num_images,), t, device=self.device).long()
            pred_noise = self.model(x, t_tensor, cond)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            if t > 0:
                noise = torch.randn_like(x)
                x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise) + torch.sqrt(beta) * noise
            else:
                x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise)
        return (x * 0.5 + 0.5).clamp(0, 1)

# Вспомогательные функции
def extract_tensors(loader):
    imgs, lbls = [], []
    for batch in loader:
        imgs.append(batch['image'])
        lbls.append(batch['label'])
    return torch.cat(imgs, dim=0), torch.cat(lbls, dim=0)

def train_classifier(model, train_loader, val_loader, epochs, model_name, lr=0.001,
                     use_focal=False, patience=3, focal_gamma=2.0):
    model = model.to(device)
    all_labels = []
    for batch in train_loader:
        if isinstance(batch, dict):
            all_labels.extend(batch['label'].cpu().numpy())
        else:
            all_labels.extend(batch[1].cpu().numpy())
    class_counts = np.bincount(all_labels)
    if len(class_counts) < 2:
        class_weights = torch.ones(2).to(device)
    else:
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * 2
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    if isinstance(model, PretrainedCNN):
        backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
        classifier_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
        optimizer = optim.Adam([{'params': backbone_params, 'lr': lr * 0.1},
                                {'params': classifier_params, 'lr': lr}])
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr*0.1, max_lr=lr, step_size_up=len(train_loader)*3)
    criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma) if use_focal else nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, dict):
                images, labels = batch['image'], batch['label']
            else:
                images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    images, labels = batch['image'], batch['label']
                else:
                    images, labels = batch[0], batch[1]
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        current_metric = val_f1 if val_f1 > 0 else val_acc
        print(f"Эпоха {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Acc: {val_acc*100:.2f}%, Val Recall: {val_recall*100:.2f}%, Val F1: {val_f1*100:.2f}%")

        if current_metric > best_val_f1:
            best_val_f1 = current_metric
            best_epoch = epoch+1
            no_improve = 0
            torch.save(model.state_dict(), f"best_{model_name}.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Ранняя остановка (лучшая метрика: {best_val_f1:.4f} на эпохе {best_epoch})")
                break

    if not os.path.exists(f"best_{model_name}.pth"):
        torch.save(model.state_dict(), f"best_{model_name}.pth")
    else:
        model.load_state_dict(torch.load(f"best_{model_name}.pth"))
    return model

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                images, labels = batch['image'], batch['label']
            else:
                images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels))>1 else 0.0
    return acc, rec, prec, f1, auc, all_labels, all_preds

def enhance_contrast_clahe(img_tensor):
    """Улучшение контраста с помощью CLAHE (только для визуализации)."""
    if not SKIMAGE_AVAILABLE:
        return img_tensor
    img_np = (img_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    if img_np.shape[-1] == 3:
        gray = np.mean(img_np, axis=2).astype(np.uint8)
        clahe = exposure.equalize_adapthist(gray, clip_limit=0.03)
        hsv = matplotlib.colors.rgb_to_hsv(img_np / 255.0)
        hsv[..., 2] = clahe
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        enhanced = (rgb * 255).astype(np.uint8)
    else:
        enhanced = (exposure.equalize_adapthist(img_np, clip_limit=0.03) * 255).astype(np.uint8)
    return torch.from_numpy(enhanced / 255.0).permute(2,0,1)

def save_comparison_image(real_loader, fake_images_tensor, save_path, label_name, apply_clahe=True):
    real_batch = next(iter(real_loader))['image'][:3]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    real_imgs = (real_batch * std + mean).clamp(0,1)
    fake_imgs = fake_images_tensor[:3].clamp(0,1)
    if apply_clahe and SKIMAGE_AVAILABLE:
        fake_imgs = torch.stack([enhance_contrast_clahe(img) for img in fake_imgs])

    fig, axes = plt.subplots(2, 3, figsize=(9,6))
    for i in range(3):
        axes[0,i].imshow(real_imgs[i].permute(1,2,0))
        axes[0,i].set_title(f"Реальный [{label_name}]")
        axes[0,i].axis('off')
        axes[1,i].imshow(fake_imgs[i].permute(1,2,0))
        axes[1,i].set_title(f"Сгенерированный [{label_name}]")
        axes[1,i].axis('off')
    plt.suptitle(f"Сравнение: Реальные vs Сгенерированные ({label_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Сравнение реальных и сгенерированных сохранено в {save_path}")

# Основной эксперимент
print("\n" + "="*80)
print("Начало эксперимента NIH Chest X-ray (Hernia vs Infiltration)")
print("="*80)

X_train, y_train = extract_tensors(train_loader)
X_val, y_val = extract_tensors(val_loader)
X_test, y_test = extract_tensors(test_loader)
train_loader_raw = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader_tensor = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader_tensor = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_counts = np.bincount(y_train.cpu().numpy())
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * 2
print(f"Веса классов: Нормальный={class_weights[0]:.2f}, Редкий={class_weights[1]:.2f}")

# 1. Baseline
print("\n--- 1. Baseline ---")
model_baseline = SimpleCNN()
model_baseline = train_classifier(model_baseline, train_loader_raw, val_loader_tensor,
                                  epochs=EPOCHS_CLS, model_name="Baseline", lr=0.001, use_focal=False)
acc, rec, prec, f1, auc, _, _ = evaluate_model(model_baseline, test_loader_tensor)
results = {"Baseline (SimpleCNN)": (acc, rec, prec, f1, auc)}

# 2. Cost-Sensitive
print("\n--- 2. Cost-Sensitive ---")
model_cost = SimpleCNN()
model_cost = train_classifier(model_cost, train_loader_raw, val_loader_tensor,
                              epochs=EPOCHS_CLS, model_name="CostSensitive", lr=0.001, use_focal=True)
acc, rec, prec, f1, auc, _, _ = evaluate_model(model_cost, test_loader_tensor)
results["Cost-Sensitive (Focal Loss + SimpleCNN)"] = (acc, rec, prec, f1, auc)

# 3. SMOTE + MLP
print("\n--- 3. SMOTE + MLP ---")
feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor.fc = nn.Identity()
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

def extract_features(loader):
    features, labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            feat = feature_extractor(imgs).cpu().numpy()
            lbls = batch['label'].cpu().numpy()
            features.append(feat)
            labels.append(lbls)
    return np.vstack(features), np.concatenate(labels)

X_train_feat, y_train_feat = extract_features(train_loader)
X_val_feat, y_val_feat = extract_features(val_loader)
X_test_feat, y_test_feat = extract_features(test_loader)

smote = SMOTE(random_state=seed)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_feat, y_train_feat)
print(f"SMOTE: {X_train_feat.shape} -> {X_train_resampled.shape}")
mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=200, random_state=seed)
mlp.fit(X_train_resampled, y_train_resampled)
y_pred_smote = mlp.predict(X_test_feat)
y_prob_smote = mlp.predict_proba(X_test_feat)[:,1]

# Сохраняем SMOTE результаты отдельно
smote_true = y_test_feat
smote_pred = y_pred_smote
smote_acc = accuracy_score(smote_true, smote_pred)
smote_rec = recall_score(smote_true, smote_pred, average='binary')
smote_prec = precision_score(smote_true, smote_pred, average='binary')
smote_f1 = f1_score(smote_true, smote_pred, average='binary')
smote_auc = roc_auc_score(smote_true, y_prob_smote)
results["SMOTE + MLP"] = (smote_acc, smote_rec, smote_prec, smote_f1, smote_auc)
print(f"SMOTE результаты: Acc={smote_acc*100:.2f}%, Rec={smote_rec*100:.2f}%, Prec={smote_prec*100:.2f}%, F1={smote_f1*100:.2f}%, AUC={smote_auc*100:.2f}%")

# 4. WGAN-GP обучение
print("\n--- 4. Обучение WGAN-GP ---")
minor_train_df = train_df[train_df['label'] == 1]
minor_train_dataset = ChestXrayDataset(minor_train_df, transform=train_transform, selected_diseases=SELECTED_DISEASES)
minor_train_loader = DataLoader(minor_train_dataset, batch_size=8, shuffle=True, num_workers=0)

netG = Generator().to(device)
netD = Discriminator().to(device)
torch.cuda.empty_cache()
netG = train_wgan_gp(netG, netD, minor_train_loader, epochs=EPOCHS_GAN)

print("\nГенерация безусловных изображений (WGAN-GP)...")
netG.eval()
syn_imgs_uncond = []
with torch.no_grad():
    for _ in range(NUM_SYNTHETIC // BATCH_SIZE + 1):
        noise = torch.randn(BATCH_SIZE, 100).to(device)
        imgs = netG(noise).cpu()
        imgs = (imgs * 0.5 + 0.5).clamp(0,1)
        syn_imgs_uncond.append(imgs)
syn_imgs_uncond = torch.cat(syn_imgs_uncond, dim=0)[:NUM_SYNTHETIC]
print(f"Сгенерировано {len(syn_imgs_uncond)} безусловных изображений")

# 5. Условный DDPM
print("\n--- 5. Обучение условного DDPM (коморбидности) ---")
ddpm = DDPMGenerator(cond_dim=len(SELECTED_DISEASES), image_size=IMG_SIZE, device=device, timesteps=DIFFUSION_TIMESTEPS)
ddpm.train(minor_train_loader, epochs=EPOCHS_DIFFUSION)

print("\nГенерация условных изображений DDPM...")
cond_samples = []
for batch in train_loader:
    cond_samples.extend(batch['comorbidities'].cpu())
syn_imgs_cond = []
for i in range(0, NUM_SYNTHETIC, 16):
    bs = min(16, NUM_SYNTHETIC - i)
    batch_cond = torch.stack(random.sample(cond_samples, bs)).to(device)
    imgs = ddpm.generate(batch_cond, bs, noise_std=0.05).cpu()
    syn_imgs_cond.append(imgs)
syn_imgs_cond = torch.cat(syn_imgs_cond, dim=0)[:NUM_SYNTHETIC]
print(f"Сгенерировано {len(syn_imgs_cond)} условных изображений")

# Визуализация сравнения (с CLAHE)
save_comparison_image(minor_train_loader, syn_imgs_cond,
                      '/kaggle/working/comparison_nih_ddpm.png', 'Hernia', apply_clahe=True)

# Подготовка аугментированных наборов
syn_labels = torch.ones(len(syn_imgs_uncond)).long()
X_aug_uncond = torch.cat([X_train, syn_imgs_uncond], dim=0)
y_aug_uncond = torch.cat([y_train, syn_labels], dim=0)
train_loader_aug_uncond = DataLoader(TensorDataset(X_aug_uncond, y_aug_uncond), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

syn_labels_cond = torch.ones(len(syn_imgs_cond)).long()
X_aug_cond = torch.cat([X_train, syn_imgs_cond], dim=0)
y_aug_cond = torch.cat([y_train, syn_labels_cond], dim=0)
train_loader_aug_cond = DataLoader(TensorDataset(X_aug_cond, y_aug_cond), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 6. WGAN-GP + SimpleCNN
print("\n--- 6. WGAN-GP + SimpleCNN ---")
modelA = SimpleCNN()
modelA = train_classifier(modelA, train_loader_aug_uncond, val_loader_tensor,
                          epochs=EPOCHS_CLS, model_name="WGAN_SimpleCNN", lr=0.001, use_focal=False)
accA, recA, precA, f1A, aucA, _, _ = evaluate_model(modelA, test_loader_tensor)
results["WGAN-GP + SimpleCNN"] = (accA, recA, precA, f1A, aucA)

# 7. WGAN-GP + ResNet-18
print("\n--- 7. WGAN-GP + ResNet-18 ---")
modelB = PretrainedCNN()
modelB = train_classifier(modelB, train_loader_aug_uncond, val_loader_tensor,
                          epochs=EPOCHS_CLS, model_name="WGAN_ResNet18", lr=0.001, use_focal=False)
accB, recB, precB, f1B, aucB, _, _ = evaluate_model(modelB, test_loader_tensor)
results["WGAN-GP + ResNet-18"] = (accB, recB, precB, f1B, aucB)

# 8. DDPM (условный) + SimpleCNN
print("\n--- 8. DDPM (коморбидности) + SimpleCNN ---")
modelE = SimpleCNN()
modelE = train_classifier(modelE, train_loader_aug_cond, val_loader_tensor,
                          epochs=EPOCHS_CLS, model_name="DDPM_SimpleCNN", lr=0.001, use_focal=False)
accE, recE, precE, f1E, aucE, _, _ = evaluate_model(modelE, test_loader_tensor)
results["DDPM (коморбидности) + SimpleCNN"] = (accE, recE, precE, f1E, aucE)

# 9. DDPM (условный) + ResNet-18 (улучш.)
print("\n--- 9. DDPM (коморбидности) + ResNet-18 (улучш.) ---")
modelF = PretrainedCNN()
modelF = train_classifier(modelF, train_loader_aug_cond, val_loader_tensor,
                          epochs=EPOCHS_CLS, model_name="DDPM_ResNet18_improved", lr=0.001,
                          use_focal=True, patience=5, focal_gamma=1.5)
accF, recF, precF, f1F, aucF, _, _ = evaluate_model(modelF, test_loader_tensor)
results["DDPM (коморбидности) + ResNet-18 (улучш.)"] = (accF, recF, precF, f1F, aucF)

# Итоговая таблица
print("\n" + "="*80)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
print("="*80)
for name, (acc, rec, prec, f1, auc) in results.items():
    print(f"{name:45} | Acc: {acc*100:.2f}% | Recall: {rec*100:.2f}% | Precision: {prec*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc*100:.2f}%")
print("="*80)

# Матрицы ошибок (исправлено, включая SMOTE)
base_res = evaluate_model(model_baseline, test_loader_tensor)
cost_res = evaluate_model(model_cost, test_loader_tensor)
wgan_simple_res = evaluate_model(modelA, test_loader_tensor)
wgan_resnet_res = evaluate_model(modelB, test_loader_tensor)
ddpm_simple_res = evaluate_model(modelE, test_loader_tensor)
ddpm_resnet_res = evaluate_model(modelF, test_loader_tensor)

smote_full_res = (smote_acc, smote_rec, smote_prec, smote_f1, smote_auc, smote_true, smote_pred)

model_info = [
    ("Baseline (SimpleCNN)", base_res),
    ("Cost-Sensitive (Focal Loss + SimpleCNN)", cost_res),
    ("SMOTE + MLP", smote_full_res),
    ("WGAN-GP + SimpleCNN", wgan_simple_res),
    ("WGAN-GP + ResNet-18", wgan_resnet_res),
    ("DDPM (коморб.) + SimpleCNN", ddpm_simple_res),
    ("DDPM (коморб.) + ResNet-18 (улучш.)", ddpm_resnet_res)
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for idx, (name, (acc_val, rec_val, prec_val, f1_val, auc_val, true, pred)) in enumerate(model_info):
    ConfusionMatrixDisplay.from_predictions(true, pred, ax=axes[idx],
                                            display_labels=['Норма', 'Грыжа'],
                                            colorbar=False)
    short_name = name.replace(' + ', '\n')
    axes[idx].set_title(f"{short_name}\nAcc={acc_val*100:.1f}% | F1={f1_val*100:.1f}%", fontsize=9)

for j in range(len(model_info), len(axes)):
    axes[j].axis('off')

plt.suptitle("Сравнение методов на NIH Chest X-ray", fontsize=14)
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrices_nih_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Эксперимент завершён ===")
