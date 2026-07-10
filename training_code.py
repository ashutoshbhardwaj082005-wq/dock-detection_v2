# ==========================================================
# Install Optuna
# ==========================================================
!pip install -q optuna

# ==========================================================
# Imports
# ==========================================================
from ultralytics import YOLO
import optuna
import pandas as pd
import gc
import torch

# ==========================================================
# Store Results
# ==========================================================
results_list = []

# ==========================================================
# Objective Function
# ==========================================================
def objective(trial):

    # -----------------------
    # Hyperparameter Search Space
    # -----------------------
    params = {

        "lr0": trial.suggest_float(
            "lr0",
            1e-4,
            5e-3,
            log=True
        ),

        "batch": trial.suggest_categorical(
            "batch",
            [8, 16, 32]
        ),

        "imgsz": trial.suggest_categorical(
            "imgsz",
            [640, 736, 832]
        ),

        "optimizer": trial.suggest_categorical(
            "optimizer",
            ["AdamW", "SGD"]
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-5,
            1e-3,
            log=True
        )
    }

    print("=" * 80)
    print(f"Trial {trial.number + 1}")
    print(params)
    print("=" * 80)

    # -----------------------
    # Load Model
    # -----------------------
    model = YOLO("yolo11s.pt")

    # -----------------------
    # Train
    # -----------------------
    model.train(

        data="/content/drive/MyDrive/Combined_Dataset/data.yaml",

        epochs=100,

        imgsz=params["imgsz"],

        batch=params["batch"],

        device=0,

        workers=2,

        optimizer=params["optimizer"],

        lr0=params["lr0"],

        weight_decay=params["weight_decay"],

        patience=10,

        cache=False,

        pretrained=True,

        verbose=False,

        project="/content/drive/MyDrive/YOLO11_Training",

        name=f"Optuna_Trial_{trial.number+1}",

        exist_ok=True
    )

    # -----------------------
    # Validation
    # -----------------------
    metrics = model.val()

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"mAP50     : {map50:.4f}")
    print(f"mAP50-95  : {map5095:.4f}")

    # -----------------------
    # Save Results
    # -----------------------
    results_list.append({

        "Trial": trial.number + 1,

        "lr0": params["lr0"],

        "batch": params["batch"],

        "imgsz": params["imgsz"],

        "optimizer": params["optimizer"],

        "weight_decay": params["weight_decay"],

        "Precision": precision,

        "Recall": recall,

        "mAP50": map50,

        "mAP50-95": map5095

    })

    # Free GPU Memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return map5095


# ==========================================================
# Create Study
# ==========================================================
study = optuna.create_study(direction="maximize")

# ==========================================================
# Start Optimization
# ==========================================================
study.optimize(
    objective,
    n_trials=10
)

# ==========================================================
# Results DataFrame
# ==========================================================
df = pd.DataFrame(results_list)

print("\n========================")
print(df)
print("========================")

# Save Results
df.to_csv(
    "/content/drive/MyDrive/YOLO11_Training/optuna_results.csv",
    index=False
)

# ==========================================================
# Best Trial
# ==========================================================
print("\nBest Trial")
print("Best mAP50-95 :", study.best_value)

print("\nBest Hyperparameters")

for key, value in study.best_params.items():
    print(f"{key} : {value}")

# ==========================================================
# Train Final Model with Best Parameters
# ==========================================================
best = study.best_params

print("\nTraining Final Model with Best Hyperparameters...")

final_model = YOLO("yolo11s.pt")

final_model.train(

    data="/content/drive/MyDrive/Combined_Dataset/data.yaml",

    epochs=100,

    imgsz=best["imgsz"],

    batch=best["batch"],

    device=0,

    workers=2,

    optimizer=best["optimizer"],

    lr0=best["lr0"],

    weight_decay=best["weight_decay"],

    patience=10,

    cache=False,

    pretrained=True,

    verbose=True,

    project="/content/drive/MyDrive/YOLO11_Training",

    name="Best_Optuna_Model",

    exist_ok=True
)

print("\nTraining Completed Successfully!")
