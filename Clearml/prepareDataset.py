import tarfile
import shutil
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
from tqdm import tqdm
from clearml import Task, Dataset


# =========================================================
# НАСТРОЙКИ
# =========================================================
ARCHIVE_NAME = "D:\Projects\ClearML\Data-science-and-ML\Tutorials\ClearML\train-00.tar.xz"   # архив лежит рядом со скриптом

TARGET_PROJECT = "FloorPlan_Restoration"
TARGET_DATASET_NAME = "FloorPlan_Ready_For_Training_v1"

IMG_SIZE = (512, 512)          # единый размер
BLUR_KERNEL = (15, 15)         # размытие для LQ
TEST_RATIO = 0.2               # 80/20
JPG_QUALITY = 85               # артефакты JPEG для LQ
SEED = 42

# Временные/итоговые папки
EXTRACT_DIR_NAME = "extracted_data"
OUTPUT_DIR_NAME = "dataset_ready_for_training"


# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================
def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def collect_image_files(root: Path):
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_ext
    ]


def ensure_structure(output_dir: Path) -> None:
    for split in ["train", "test"]:
        (output_dir / split / "HQ").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "LQ").mkdir(parents=True, exist_ok=True)


def make_unique_name(file_path: Path) -> str:
    # Чтобы не было конфликтов имён при одинаковых stem в разных папках
    rel = str(file_path).replace("\\", "_").replace("/", "_")
    stem = Path(rel).stem
    return stem


def process_one(job):
    """
    job = (file_path_str, split, output_dir_str, img_size, blur_kernel, jpg_quality)
    """
    file_path_str, split, output_dir_str, img_size, blur_kernel, jpg_quality = job

    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)

    img = cv2.imread(str(file_path))
    if img is None:
        return {"ok": False, "file": file_path_str, "reason": "cv2.imread returned None"}

    # HQ = resize
    hq = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    # LQ = blur
    lq = cv2.GaussianBlur(hq, blur_kernel, 0)

    # Дополнительные jpeg-артефакты
    ok, encoded = cv2.imencode(
        ".jpg",
        lq,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    )
    if not ok:
        return {"ok": False, "file": file_path_str, "reason": "cv2.imencode failed"}

    lq = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if lq is None:
        return {"ok": False, "file": file_path_str, "reason": "cv2.imdecode failed"}

    base_name = make_unique_name(file_path)

    hq_out = output_dir / split / "HQ" / f"{base_name}.png"
    lq_out = output_dir / split / "LQ" / f"{base_name}.jpg"

    ok_hq = cv2.imwrite(str(hq_out), hq)
    ok_lq = cv2.imwrite(str(lq_out), lq)

    if not ok_hq or not ok_lq:
        return {"ok": False, "file": file_path_str, "reason": "cv2.imwrite failed"}

    return {"ok": True, "file": file_path_str}


# =========================================================
# ОСНОВНОЙ КОД
# =========================================================
def main():
    # 1. Init ClearML task
    task = Task.init(
        project_name=TARGET_PROJECT,
        task_name="Prepare_FloorPlan_Ready_For_Training_Local",
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=False
    )

    script_dir = Path(__file__).resolve().parent
    archive_path = script_dir / ARCHIVE_NAME
    extract_dir = script_dir / EXTRACT_DIR_NAME
    output_dir = script_dir / OUTPUT_DIR_NAME

    print(f"Рабочая папка: {script_dir}")
    print(f"Архив: {archive_path}")

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Не найден архив '{ARCHIVE_NAME}' рядом со скриптом: {archive_path}"
        )

    # 2. Очистка старых папок
    print("Шаг 1: очистка временных папок...")
    safe_rmtree(extract_dir)
    safe_rmtree(output_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_structure(output_dir)

    # 3. Распаковка
    print(f"Шаг 2: распаковка архива {ARCHIVE_NAME}...")
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(path=extract_dir)

    # 4. Сбор изображений
    print("Шаг 3: поиск изображений...")
    all_files = collect_image_files(extract_dir)
    print(f"Найдено изображений: {len(all_files)}")

    if not all_files:
        raise RuntimeError("После распаковки изображения не найдены")

    # 5. Train/Test split
    print("Шаг 4: train/test split...")
    random.seed(SEED)
    random.shuffle(all_files)

    split_idx = int(len(all_files) * (1 - TEST_RATIO))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    print(f"Train: {len(train_files)}")
    print(f"Test:  {len(test_files)}")

    # 6. Подготовка jobs
    print("Шаг 5: генерация HQ/LQ...")
    jobs = []
    for p in train_files:
        jobs.append((str(p), "train", str(output_dir), IMG_SIZE, BLUR_KERNEL, JPG_QUALITY))
    for p in test_files:
        jobs.append((str(p), "test", str(output_dir), IMG_SIZE, BLUR_KERNEL, JPG_QUALITY))

    workers = max(1, cpu_count() - 1)
    print(f"Используем процессов: {workers}")

    success_count = 0
    failed = []

    with Pool(processes=workers) as pool:
        for result in tqdm(pool.imap_unordered(process_one, jobs, chunksize=32), total=len(jobs)):
            if result["ok"]:
                success_count += 1
            else:
                failed.append(result)

    print(f"Успешно обработано: {success_count}")
    print(f"Ошибок: {len(failed)}")

    if failed:
        print("Примеры ошибок:")
        for item in failed[:10]:
            print(item)

    # 7. Логируем параметры в ClearML
    task.connect({
        "archive_name": ARCHIVE_NAME,
        "target_project": TARGET_PROJECT,
        "target_dataset_name": TARGET_DATASET_NAME,
        "img_size": IMG_SIZE,
        "blur_kernel": BLUR_KERNEL,
        "test_ratio": TEST_RATIO,
        "jpg_quality": JPG_QUALITY,
        "seed": SEED,
        "processed_ok": success_count,
        "processed_failed": len(failed),
    })

    # 8. Загрузка в ClearML Dataset
    print("Шаг 6: регистрация итогового датасета в ClearML...")
    ds = Dataset.create(
        dataset_project=TARGET_PROJECT,
        dataset_name=TARGET_DATASET_NAME
    )

    ds.add_files(path=str(output_dir))
    ds.upload()
    ds.finalize()

    print("✅ Готово!")
    print(f"Dataset ID: {ds.id}")
    print(f"Структура датасета: {output_dir}")
    print("Дальше его можно напрямую использовать в PyTorch Dataset/DataLoader.")


if __name__ == "__main__":
    main()