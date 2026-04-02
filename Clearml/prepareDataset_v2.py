import sys
from pathlib import Path

from clearml import Task, Dataset

# Для Windows-агента
if sys.platform == "win32":
    Task.ignore_requirements("pywin32")

task = Task.init(
    project_name="FloorPlan_Restoration",
    task_name="Tensor_Preprocessing_Final_v3",
    task_type=Task.TaskTypes.data_processing,
    reuse_last_task_id=False
)

# Не передаем pip-опции внутрь package name
task.set_packages([
    "torch==2.3.1",
    "opencv-python==4.13.0.92",
    "tqdm==4.67.3",
    "numpy"
])

# Отправляем задачу на агент
task.execute_remotely(queue_name="default")

# ----- код ниже выполняется уже на агенте -----
import cv2
import torch
from tqdm import tqdm

SOURCE_PROJECT = "FloorPlan_Restoration"
SOURCE_NAME = "FloorPlan_Processed_v2"
TARGET_NAME = "FloorPlan_Final_Tensors"
IMG_SIZE = (512, 512)

print(f"Шаг 1: Загрузка датасета {SOURCE_NAME}...")
input_ds = Dataset.get(
    dataset_project=SOURCE_PROJECT,
    dataset_name=SOURCE_NAME
)
input_path = Path(input_ds.get_local_copy())

processed_dir = Path("./tensors_out")
for split in ["HQ", "LQ"]:
    (processed_dir / split).mkdir(parents=True, exist_ok=True)

print("Шаг 2: Конвертация изображений в тензоры...")

# ищем либо train/HQ, train/LQ, либо просто HQ, LQ
for split in ["HQ", "LQ"]:
    current_folder = input_path / "train" / split
    if not current_folder.exists():
        current_folder = input_path / split

    if not current_folder.exists():
        print(f"Папка для split={split} не найдена, пропускаю")
        continue

    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
        files.extend(current_folder.glob(ext))

    for f in tqdm(files, desc=f"Processing {split}"):
        img = cv2.imread(str(f))
        if img is None:
            continue

        # 1. BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. resize
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

        # 3. [0,1], float32, CHW
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0

        # сохраняем .pt
        torch.save(tensor, processed_dir / split / f"{f.stem}.pt")

print("Шаг 3: Регистрация нового датасета в ClearML...")
new_ds = Dataset.create(
    dataset_project=SOURCE_PROJECT,
    dataset_name=TARGET_NAME,
    parent_datasets=[input_ds.id]
)
new_ds.add_files(path=str(processed_dir))
new_ds.upload()
new_ds.finalize()

print("✅ Готово! Новый датасет зарегистрирован в ClearML.")