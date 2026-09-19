import torch
from ultralytics import RTDETR

torch.cuda.reset_peak_memory_stats()
RTDETR("rtdetr-l.pt").train(
    data="coco8.yaml", epochs=1, imgsz=640, batch=4, device=0,
    workers=0,          # Windows: avoid dataloader worker spawn issues
    project="runs/smoke", name="rtdetr_b4", exist_ok=True,
)
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")