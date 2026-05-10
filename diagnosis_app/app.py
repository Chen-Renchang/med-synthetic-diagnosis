# -*- coding: utf-8 -*-
"""
Прототип системы диагностики (Gradio)
Использует обученную модель DDPM + ResNet-18 (NIH: Hernia vs Infiltration).
Загружает модель, предобрабатывает входной снимок и выдаёт вероятности классов.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr


# Определение модели классификатора (PretrainedCNN)
class PretrainedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# Настройки
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
MODEL_PATH = "./best_DDPM_ResNet18_improved.pth"  # укажите правильный путь
CLASS_NAMES = ['Норма (Infiltration)', 'Патология (Hernia)']

# Загрузка модели
model = PretrainedCNN(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Предобработка изображения
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#  Функция предсказания
def predict(image: Image.Image):
    """Принимает PIL Image, возвращает словарь с вероятностями классов."""
    img = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]  # [prob_class0, prob_class1]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(2)}


#  Интерфейс Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите рентгеновский снимок"),
    outputs=gr.Label(num_top_classes=2, label="Результат диагностики"),
    title="Система поддержки диагностики редких патологий",
    description="Прототип на основе диффузионной модели DDPM + ResNet-18. "
                "Загрузите изображение грудной клетки для оценки вероятности Hernia.",
    examples=[["example_normal.png"], ["example_hernia.png"]]  # при необходимости добавить примеры
)

if __name__ == "__main__":
    iface.launch(share=False)  # share=True для временной публичной ссылки
