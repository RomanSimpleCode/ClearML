import os
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from clearml import Task, Dataset

# --- НАСТРОЙКИ ---
SOURCE_PROJECT = "datasets"
SOURCE_NAME = "FloorPlanDataset"
TARGET_PROJECT = "FloorPlan_Restoration"
TARGET_NAME = "FloorPlan_Processed_v2"

# Параметры обработки
IMG_SIZE = (512, 512)   # Размер для обучения
BLUR_KERNEL = (15, 15)  # Степень нечеткости (чем больше числа, тем сильнее размытие)
TEST_SIZE = 0.2
JPG_QUALITY = 85        # Качество JPG для LQ (добавляет легкие артефакты сжатия)

def check_dataset_exists(project, name):
    datasets = Dataset.list_datasets(
        dataset_project=project,
        dataset_name=name,
        only_completed=True
    )
    return len(datasets) > 0

# 1. Инициализация задачи
task = Task.init(
    project_name=TARGET_PROJECT, 
    task_name="Dataset_HQ_PNG_LQ_JPG",
    task_type=Task.TaskTypes.data_processing
)

# 2. Проверка исходного датасета
print(f"Ищем датасет {SOURCE_NAME}...")
if not check_dataset_exists(SOURCE_PROJECT, SOURCE_NAME):
    print(f"ОШИБКА: Исходный датасет '{SOURCE_NAME}' в проекте '{SOURCE_PROJECT}' не найден.")
    task.close()
    exit(1)

# 3. Получение данных
input_dataset = Dataset.get(dataset_project=SOURCE_PROJECT, dataset_name=SOURCE_NAME)
input_path = Path(input_dataset.get_local_copy())

# Собираем файлы
all_files = [f for f in input_path.rglob('*') if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
print(f"Найдено файлов: {len(all_files)}")

# Разделение на выборки
train_files, test_files = train_test_split(all_files, test_size=TEST_SIZE, random_state=42)

# 4. Обработка
temp_dir = Path("./processed_data_temp")

for split, files in [("train", train_files), ("test", test_files)]:
    print(f"Обработка {split}...")
    
    # Создаем папки
    hq_path = temp_dir / split / "HQ"
    lq_path = temp_dir / split / "LQ"
    hq_path.mkdir(parents=True, exist_ok=True)
    lq_path.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(files):
        img = cv2.imread(str(file_path))
        if img is None:
            continue
        
        # 1. Ресайз (базовая обработка)
        hq = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        
        # 2. Создание нечеткого изображения (только блюр, без шума)
        lq = cv2.GaussianBlur(hq, BLUR_KERNEL, 0)
        
        # 3. Сохранение HQ в PNG
        base_name = file_path.stem
        cv2.imwrite(str(hq_path / f"{base_name}.png"), hq)
        
        # 4. Сохранение LQ в JPG (нечеткое)
        cv2.imwrite(
            str(lq_path / f"{base_name}.jpg"), 
            lq, 
            [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]
        )

# 5. Создание нового датасета в ClearML
print("Регистрация нового датасета в ClearML...")
new_ds = Dataset.create(
    dataset_project=TARGET_PROJECT,
    dataset_name=TARGET_NAME,
    parent_datasets=[input_dataset.id]
)

new_ds.add_files(path=str(temp_dir))
new_ds.upload()
new_ds.finalize()

print(f"Готово! ID нового датасета: {new_ds.id}")