from pathlib import Path
import torch
CONFIG = {
    # 随机性与数据切分
    "random_state": 42, 
    "num_boost_round": 1000, 
    "test_size": 0.20,
    # 全量口径起点：2017 年 8 月数据仅用于初期构造滑动窗口等特征工程，不进入模型与统计
    "full_start_date": "2017-09-01",
    # 非对称 Pseudo-Huber 损失
    "under_weights": [5.0, 20.0, 40.0],
    "over_weight": 1.0,
    "delta": 0.2,  # delta 作用于对数配送时长空间
    # 超时风险评估
    "deadline_days": 10.0,
    "severe_under_days": 7.0,
    # LightGBM 参数
    "learning_rate": 0.03, 
    "num_leaves": 31, 
    "max_depth": -1, 
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8, 
    "bagging_fraction": 0.8, 
    "bagging_freq": 1,
    "lambda_l1": 0.0, 
    "lambda_l2": 2.0, 
    "min_sum_hessian_in_leaf": 1e-3, 
    "num_threads": -1,
    # MLP 参数（二阶段超时分类器）
    'input_dim' : 157,      # 占位值，运行时按特征矩阵实际宽度覆盖
    'device' : 'cuda:0',    # 无可用 GPU 时自动退回 CPU
    'hidden_dim' : 256,
    'output_dim' : 2,
    # MLP 训练参数：AdamW + 线性 warmup + 余弦退火
    "mlp_epochs": 40,
    "mlp_batch_size": 512,
    "mlp_lr": 1e-3,
    "mlp_weight_decay": 1e-4,   # AdamW 的权重衰减
    "mlp_warmup_epochs": 3,     # 线性 warmup 轮数，warmup 结束后接余弦退火
    "mlp_min_lr": 1e-5,         # 余弦退火的学习率下界
    "mlp_class_weight": True,   # 按训练集类别频率加权，缓解正负样本不平衡
    "mlp_log_interval": 5,      # 每多少轮打印一次训练日志
    # 训练/测试切分方式：time = 按时间后 20%（与回归模型同口径），random = 分层随机 20%
    "mlp_split": "time",
    # 时间序列 K 折：产出每条订单的样本外（OOF）预测值，用于构造二阶段的「是否超时」标签。
    # 最早的那一段没有更早的数据可训练，不产生 OOF 预测，分类器会自动跳过
    "oof_folds": 5,

    # 图表设置
    "figure_size": (12, 6), "figure_dpi": 200,
    
    # 输出路径
    "model_dir": Path(__file__).parent.parent / "model",
    "asset_dir": Path(__file__).parent.parent / "assets",
    'report_dir': Path(__file__).parent.parent / "report",
    'output_dir': Path(__file__).parent.parent / "output",
    'base_dir':  Path(__file__).parent.parent

    
}

