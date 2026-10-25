# %% 1. 导入库
from pathlib import Path
from config import CONFIG

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from data_prepare import df as feature_data  # R 构造好的特征表（与 lgb_model.py 同源）

BASE_DIR = CONFIG['base_dir']
OUTPUT_DIR = Path(CONFIG['output_dir'])
FULL_START_TIME = pd.Timestamp(CONFIG['full_start_date'])
RANDOM_STATE = CONFIG['random_state']
# 配置里写的是 cuda:0，没有可用 GPU 时自动退回 CPU
DEVICE = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
print(f"计算设备：{DEVICE}")

# %% 2. 读取全量预测数据（lgb_val.py 的产出，72907 行）
df = pd.read_csv(OUTPUT_DIR / "LightGBM预测数据.csv", encoding="utf-8-sig")
df['下单时间'] = pd.to_datetime(df['下单时间'], format='mixed')

# %% 3. 行对齐：并入 R 构造的工程特征
# 预测数据表里只有原始订单字段，工程特征在 processed_data/LightGBM.csv 里。
# 行对齐规则与 lgb_val.py 完全一致：先取「下单时间 >= 最小时间 + 30 天」的行，
# 再按 发货州 / 收货州 / 下单时间 / seller_id 做稳定排序复现特征表的行次序。
orders = pd.read_csv(BASE_DIR / "data" / "订单数据.csv", encoding="utf-8-sig")
orders['下单时间'] = pd.to_datetime(orders['下单时间'], format='mixed')

warm_mask = orders['下单时间'] >= orders['下单时间'].min() + pd.Timedelta(days=30)
full_mask = orders['下单时间'] >= FULL_START_TIME
keep_index = orders.index[warm_mask & full_mask]  # 预测数据表的 72907 行

aligned_index = (orders.loc[warm_mask]
                 .sort_values(["发货州", "收货州", "下单时间", "seller_id"], kind="stable").index)
if len(aligned_index) != len(feature_data):
    raise ValueError(f"行数对不上：订单数据筛出 {len(aligned_index)} 行，特征表有 {len(feature_data)} 行。")

feat = feature_data.copy()
feat['_row_id'] = np.arange(len(feat))            # 特征表原始行号，用于并入 OOF 预测值
feat.index = aligned_index
feat = feat.loc[feat.index.intersection(keep_index)]  # 去掉 8 月预热订单

if len(feat) != len(df):
    raise ValueError(f"行对齐失败：特征表 {len(feat)} 行，预测数据 {len(df)} 行。")

# 预测数据表就是订单表按 keep_index 取出的行，行序一致，因此可以直接按行号拼接
df.index = keep_index
data = feat.join(df[['下单时间', '总运送周期'] + [c for c in df.columns if '预期总周期' in c]])

# 校验：两表拼起来后，真实总运送周期必须逐行相同
if not np.isclose(data["total_delivery_time"], data["总运送周期"], atol=1e-6, equal_nan=True).all():
    raise ValueError("行对齐校验失败：total_delivery_time 与 总运送周期 不一致。")
print(f"工程特征并入完成：{len(data)} 行，行对齐校验通过")

# %% 4. 并入样本外（OOF）预测值，构造「是否超时」标签
# lgb_val.py 回填的预测值里，前 80% 是各模型自己的训练样本，属于样本内预测；
# 用样本内残差定义标签会让分类器学到「模型在这批见过的单上拟合偏了没有」，无法迁移到新订单。
# 因此这里改用 lgb_model.py 第 12 节时间序列 K 折产出的 OOF 预测值来定义标签。
oof = pd.read_csv(OUTPUT_DIR / "oof_predictions.csv", encoding="utf-8-sig")
oof_cols = [c for c in oof.columns if c.endswith('_oof_预期总周期')]
has_oof = oof[oof_cols].notna().all(axis=1)
data = data.merge(oof.loc[has_oof, ['_row_id'] + oof_cols], on='_row_id', how='inner')

print(f"\n有 OOF 预测的订单：{len(data)} 条（最早一段 {int((~has_oof).sum())} 条没有更早的数据可训练，已剔除）")

# 判断口径与各模型的延迟率一致：预期总周期 < 实际总运送周期，即承诺比实际短、该单会超时
data['L1是否超时'] = (data['L1_oof_预期总周期'] < data['总运送周期']).astype('int8')
data['5x是否超时'] = (data['PseudoHuber_Under5x_oof_预期总周期'] < data['总运送周期']).astype('int8')
# 样本内口径仅留作对照，不作为训练目标
data['L1是否超时_样本内'] = (data['L1预期总周期'] < data['总运送周期']).astype('int8')
data['5x是否超时_样本内'] = (data['PseudoHuber_Under5x预期总周期'] < data['总运送周期']).astype('int8')

for col in ['L1', '5x']:
    print(f"{col}是否超时（OOF 口径）：正样本 {int(data[col + '是否超时'].sum())} 条，"
          f"占比 {data[col + '是否超时'].mean():.2%} | 样本内口径对照 {data[col + '是否超时_样本内'].mean():.2%}")

# %% 5. 特征矩阵
# 特征选择与 lgb_model.py 保持一致：剔除目标变量、订单标识和高基数字段；
# 类别变量只保留州级与商品/付款方式，城市与 seller_id 基数过高，不做 one-hot。
EXCLUDE_COLS = ["order_time", "seller_id", "seller_city", "customer_city", "_row_id",
                "total_delivery_time", "log_delivery_time", "estimated_delivery_time"]
CAT_FEATURES = ["product_type", "payment_method", "seller_state", "customer_state"]
# 四个模型的样本外预估：下单时刻即可算出，不含实际送达结果，不算数据泄露
MODEL_FEATURES = oof_cols
NUM_FEATURES = [c for c in feat.columns if c not in EXCLUDE_COLS + CAT_FEATURES]

X_cat = pd.get_dummies(data[CAT_FEATURES].astype("object").fillna("未知"), dtype=float)
X = pd.concat([data[NUM_FEATURES + MODEL_FEATURES].astype(float), X_cat], axis=1)
INPUT_DIM = X.shape[1]

print(f"\n特征矩阵：{INPUT_DIM} 维 = 数值特征 {len(NUM_FEATURES)} + 模型样本外预估 {len(MODEL_FEATURES)}"
      f" + 类别 one-hot {X_cat.shape[1]}")
print(f"数值特征：{NUM_FEATURES}")
print(f"模型样本外预估特征：{MODEL_FEATURES}")
print(f"类别特征：{CAT_FEATURES}")

# 切分方式：time = 按时间后 20%（与回归模型同口径，测试集是更晚的订单）；
# random = 分层随机抽取 20%，只用于检验特征与标签的可学性，结果偏乐观
labels_all = data["L1是否超时"].to_numpy(dtype=np.int64)
if CONFIG['mlp_split'] == "time":
    cutoff_time = data["order_time"].quantile(1 - CONFIG['test_size'])
    train_mask = (data["order_time"] < cutoff_time).to_numpy()
    print(f"切分方式：时间切分，切分点 {cutoff_time}")
else:
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=CONFIG['test_size'],
                                           random_state=RANDOM_STATE, stratify=labels_all)
    train_mask = np.zeros(len(data), dtype=bool)
    train_mask[train_idx] = True
    print("切分方式：分层随机切分（注意：随机切分的指标偏乐观，不能代表上线效果）")
test_mask = ~train_mask

# 缺失值用训练集的中位数填补，再做标准化，避免测试集信息流入训练
X_all = X.to_numpy(dtype=np.float64)
medians = np.nanmedian(X_all[train_mask], axis=0)
X_all = np.where(np.isnan(X_all), medians, X_all)
mean = X_all[train_mask].mean(axis=0)
std = X_all[train_mask].std(axis=0)
std[std < 1e-8] = 1.0
X_all = ((X_all - mean) / std).astype(np.float32)

print(f"训练集 {int(train_mask.sum())} 条 | 测试集 {int(test_mask.sum())} 条")

# %% 6. MLP 模型定义
class MLP(nn.Module):
    def __init__(self, input_dim=CONFIG['input_dim'], hidden_dim=CONFIG['hidden_dim'], output_dim=CONFIG['output_dim']):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, logits=False):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # 训练时返回 logits：CrossEntropyLoss 内部自带 log_softmax，
        # 再叠一层 Softmax 会把概率压扁、梯度变小，所以只在推理时取概率
        if logits:
            return out
        return self.softmax(out)


# %% 7. 训练与评估
def train_and_evaluate(X, y, label_name):
    """用 MLP 对「是否超时」做监督学习，评估指标按切分出的测试集给出。"""
    X_train, y_train = torch.tensor(X[train_mask]), torch.tensor(y[train_mask], dtype=torch.long)
    X_test, y_test = torch.tensor(X[test_mask]), torch.tensor(y[test_mask], dtype=torch.long)

    print(f"\n{'=' * 60}\n目标变量：{label_name}")
    print(f"训练集正样本占比 {y_train.float().mean():.2%} | 测试集正样本占比 {y_test.float().mean():.2%}")

    # 类别权重：按训练集的类别频率做均衡加权
    class_counts = np.bincount(y_train.numpy(), minlength=CONFIG['output_dim']).astype(float)
    if CONFIG['mlp_class_weight']:
        class_weight = torch.tensor(class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1)),
                                    dtype=torch.float32, device=DEVICE)
    else:
        class_weight = None
    print(f"类别权重：{class_weight.tolist() if class_weight is not None else '不加权'}")

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=CONFIG['mlp_batch_size'], shuffle=True, drop_last=False)

    model = MLP(input_dim=INPUT_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['mlp_lr'],
                                  weight_decay=CONFIG['mlp_weight_decay'])

    # 学习率调度：先线性 warmup，再余弦退火
    warmup_epochs = CONFIG['mlp_warmup_epochs']
    epochs = CONFIG['mlp_epochs']
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=CONFIG['mlp_min_lr'])
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    X_test_dev, y_test_dev = X_test.to(DEVICE), y_test.to(DEVICE)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb, logits=True), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        scheduler.step()

        if epoch == 1 or epoch % CONFIG['mlp_log_interval'] == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test_dev, logits=True), y_test_dev).item()
                prob = model(X_test_dev)[:, 1].cpu().numpy()
            print(f"epoch {epoch:3d}/{epochs} | lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"train_loss {epoch_loss / len(y_train):.4f} | test_loss {test_loss:.4f} | "
                  f"test_auc {roc_auc_score(y_test.numpy(), prob):.4f}")

    # 测试集指标
    model.eval()
    with torch.no_grad():
        prob = model(X_test_dev)[:, 1].cpu().numpy()
    y_true = y_test.numpy()
    y_pred = (prob >= 0.5).astype(int)

    print(f"\n---------- {label_name}：测试集分类效果 ----------")
    print(f"准确率 {accuracy_score(y_true, y_pred):.4f} | AUC {roc_auc_score(y_true, prob):.4f}")
    print(f"正类（超时）精确率 {precision_score(y_true, y_pred, zero_division=0):.4f} | "
          f"召回率 {recall_score(y_true, y_pred, zero_division=0):.4f} | "
          f"F1 {f1_score(y_true, y_pred, zero_division=0):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"混淆矩阵：真负 {tn} | 假正 {fp} | 假负 {fn} | 真正 {tp}")

    return ({"label": label_name, "accuracy": accuracy_score(y_true, y_pred),
             "auc": roc_auc_score(y_true, prob),
             "precision": precision_score(y_true, y_pred, zero_division=0),
             "recall": recall_score(y_true, y_pred, zero_division=0),
             "f1": f1_score(y_true, y_pred, zero_division=0)}, prob)


# %% 8. 分别训练 L1、5x 两个分类器（各自独立训练，不共享参数）
results, probs = {}, {}
for label_col in ["L1是否超时", "5x是否超时"]:
    results[label_col], probs[label_col] = train_and_evaluate(
        X_all, data[label_col].to_numpy(dtype=np.int64), label_col)

print(f"\n{'=' * 60}\n========== 两个分类器的测试集效果汇总 ==========")
print(pd.DataFrame(results.values()).round(4).to_string(index=False))

# %% 9. 二阶段路由效果：低风险订单走 L1，高风险订单走保守模型
# 分类器真正的价值要看路由之后的延迟率和预期时长，而不是单看 AUC。
# 这里用 L1 分类器的输出概率做门控：概率越高说明 L1 的预估越可能偏紧，改走保守模型。
actual_test = data.loc[test_mask, "总运送周期"].to_numpy(dtype=float)
pred_test = {name: data.loc[test_mask, f"{name}_oof_预期总周期"].to_numpy(dtype=float)
             for name in ["L1", "PseudoHuber_Under5x", "PseudoHuber_Under20x", "PseudoHuber_Under40x"]}


def routing_metrics(final_pred):
    error = final_pred - actual_test
    return {"延迟率": np.mean(final_pred < actual_test),
            "平均预期总周期": np.mean(final_pred),
            "MAE": np.mean(np.abs(error))}


routing_rows = []
for name in pred_test:
    routing_rows.append({"策略": f"全部使用{name}", **routing_metrics(pred_test[name])})

risk_prob = probs["L1是否超时"]
for conservative in ["PseudoHuber_Under5x", "PseudoHuber_Under20x"]:
    for threshold in [0.3, 0.5, 0.7]:
        use_conservative = risk_prob >= threshold
        final = np.where(use_conservative, pred_test[conservative], pred_test["L1"])
        routing_rows.append({
            "策略": f"路由(L1→{conservative.replace('PseudoHuber_Under', '')}, 阈值{threshold})",
            "保守模型使用占比": use_conservative.mean(),
            **routing_metrics(final),
        })

routing_df = pd.DataFrame(routing_rows)
print(f"\n========== 测试集（{len(actual_test)} 条）二阶段路由效果 ==========")
print("延迟率：预期总周期 < 真实总运送周期的订单占比，越低越好。")
print(routing_df.round(4).to_string(index=False))
routing_df.to_csv(OUTPUT_DIR / "stage2_routing_comparison.csv", index=False, encoding="utf-8-sig")
print(f"\n路由效果已保存：{OUTPUT_DIR / 'stage2_routing_comparison.csv'}")
