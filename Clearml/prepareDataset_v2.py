import sys
from clearml import Task, Dataset

# 1. Настройка для удаленного запуска
if sys.platform == "win32":
    Task.ignore_requirements("pywin32")

task = Task.init(
    project_name="FloorPlan_Restoration", 
    task_name="Tensor_Preprocessing_Final",
    task_type=Task.TaskTypes.data_processing
)

# Выполняем на агенте
task.execute_remotely(queue_name='default')

# Явно просим агента поставить библиотеки
task.set_packages(["torch", "opencv-python", "tqdm", "numpy"])

import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# --- ПАРАМЕТРЫ ---
# БЕРЕМ ТВОЙ ВТОРОЙ ДАТАСЕТ КАК ИСТОЧНИК
SOURCE_PROJECT = "FloorPlan_Restoration"
SOURCE_NAME = "FloorPlan_Processed_v2" 
TARGET_NAME = "FloorPlan_Final_Tensors"
IMG_SIZE = (512, 512)

print(f"Шаг 1: Загрузка промежуточного датасета {SOURCE_NAME}...")
input_ds = Dataset.get(dataset_project=SOURCE_PROJECT, dataset_name=SOURCE_NAME)
input_path = Path(input_ds.get_local_copy())

# Создаем структуру для тензоров
processed_dir = Path("./tensors_out")
for d in ["HQ", "LQ"]: (processed_dir / d).mkdir(parents=True, exist_ok=True)

# Теперь мы ищем картинки в папках HQ и LQ нашего промежуточного датасета
print("Шаг 2: Конвертация картинок в тензоры...")

for split in ["HQ", "LQ"]:
    current_folder = input_path / "train" / split
    # Если структуры 'train' нет, ищем просто в корне папки split
    if not current_folder.exists():
        current_folder = input_path / split
        
    files = list(current_folder.glob("*.p*g")) + list(current_folder.glob("*.j*g"))
    
    for f in tqdm(files, desc=f"Processing {split}"):
        img = cv2.imread(str(f))
        if img is None: continue
        
        # 1) BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2) Нормализация [0,1] float32 & 3) PyTorch (CHW)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Сохраняем (имя файла оставляем то же, меняем расширение на .pt)
        torch.save(tensor, processed_dir / split / f"{f.stem}.pt")

# Шаг 3: Создание финального датасета
print("Шаг 3: Регистрация тензорного датасета...")
new_ds = Dataset.create(
    dataset_project="FloorPlan_Restoration",
    dataset_name=TARGET_NAME,
    parent_datasets=[input_ds.id]
)
new_ds.add_files(path=str(processed_dir))
new_ds.upload()
new_ds.finalize()

print(f"✅ Готово! Теперь у нас есть датасет тензоров: {TARGET_NAME}")