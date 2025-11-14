import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from scipy import stats

# === 配置路径 ===
base_path_ours = "/mnt/sdd/wjh/Fei_Bing/model_kfold_Xi_Ru"
# 如果有第二个模型（用于显著性检验），例如 baseline：
base_path_baseline = "/mnt/sdd/wjh/Fei_Bing/model_kfold_baseline"  # 可留空

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'macro')
    sens = recall_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'macro')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) == 2 else [0,0,0,0]
    spec = tn / (tn + fp + 1e-8) if tn + fp > 0 else 0
    return acc, prec, sens, spec

def analyze_model(base_path):
    metrics = []
    for fold in range(5):
        fold_path = os.path.join(base_path, f"fold{fold}/preds")
        y_true = np.load(os.path.join(fold_path, f"y_true_fold{fold}.npy"))
        y_pred = np.load(os.path.join(fold_path, f"y_pred_fold{fold}.npy"))
        acc, prec, sens, spec = compute_metrics(y_true, y_pred)
        metrics.append([acc, prec, sens, spec])
        print(f"Fold {fold}: Acc={acc:.4f}, Prec={prec:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")
    metrics = np.array(metrics)
    mean = metrics.mean(axis=0)
    std = metrics.std(axis=0)
    ci95_low = mean - 1.96 * std / np.sqrt(len(metrics))
    ci95_high = mean + 1.96 * std / np.sqrt(len(metrics))
    print("\n=== 平均结果 ± 标准差 ===")
    for i, name in enumerate(["Acc", "Prec", "Sens", "Spec"]):
        print(f"{name}: {mean[i]:.4f} ± {std[i]:.4f} (95% CI: [{ci95_low[i]:.4f}, {ci95_high[i]:.4f}])")
    return metrics

# === 主流程 ===
print("==== 分析我方模型 ====")
ours_metrics = analyze_model(base_path_ours)

if os.path.exists(base_path_baseline):
    print("\n==== 分析基线模型 ====")
    baseline_metrics = analyze_model(base_path_baseline)

    print("\n==== 显著性检验（paired t-test） ====")
    for i, name in enumerate(["Acc", "Prec", "Sens", "Spec"]):
        t_stat, p_value = stats.ttest_rel(ours_metrics[:, i], baseline_metrics[:, i])
        print(f"{name}: t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            print(f"→ {name} 差异显著 (p < 0.05)")
        else:
            print(f"→ {name} 差异不显著")
