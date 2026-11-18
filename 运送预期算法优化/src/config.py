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
    'device' : 'auto',      # auto=有CUDA则用CUDA，否则CPU；可显式写 cuda:0 / cpu（卡不存在时自动回退）
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

    # ============================================================
    # Transformer + VCN 对比模型（Transformer_CNN_model*.py）
    # 思路：Transformer 编码平台物流状态的长期演化（长时序依赖），
    #       多尺度 1D 卷积 VCN 捕捉近几个时间片的局部突变（局部感受野），
    #       二者融合后再拼接订单当期静态特征，输出对数配送周期。
    # ============================================================
    # 序列构造：把历史切成 time_steps 个时间桶，每桶 bucket_days 天；第 i 单的历史 =
    # 从它所在桶往前数 time_steps 个已结束的桶，每桶聚合成 seq_feature_dim 维快照。
    "tc_seq_steps": 16,          # 序列长度（历史时间片个数）
    "tc_bucket_days": 3,         # 每个时间片的跨度（天），16×3≈48 天历史窗口
    "tc_seq_mode": "global",     # global=全平台快照(稠密)；route=按发货州→收货州聚合(更稀疏)
    # Transformer 编码器
    "tc_d_model": 64,            # 序列 embedding / 注意力维度
    "tc_nhead": 4,               # 多头注意力头数（须能整除 d_model）
    "tc_num_layers": 2,          # TransformerEncoder 层数
    "tc_dim_feedforward": 128,   # 前馈网络隐层维度
    "tc_dropout": 0.1,
    # VCN：多尺度一维卷积，核越大感受野越宽（看最近若干时间片）
    "tc_vcn_channels": 32,       # 每个卷积核输出的通道数
    "tc_vcn_kernel_sizes": [3, 5, 7],
    # 静态特征分支与融合
    "tc_static_hidden": 64,
    "tc_fusion_hidden": 64,
    # 训练：AdamW + 线性 warmup + 余弦退火（与 MLP 分类器同策略）
    "tc_epochs": 100,
    "tc_batch_size": 256,
    "tc_lr": 1e-3,
    "tc_weight_decay": 1e-4,
    "tc_warmup_epochs": 5,
    "tc_min_lr": 1e-5,
    "tc_grad_clip": 1.0,         # 梯度裁剪，稳定Transformer训练
    "tc_log_interval": 5,
    # 损失：L1 基线 + 与 LightGBM 同款非对称 Pseudo-Huber（低估惩罚加倍）
    "tc_loss": "l1",             # 仅作默认展示；实际按 tc_under_weights 自动生成各模型
    "tc_under_weights": [5.0, 20.0, 40.0],
    "tc_delta": 0.2,             # Pseudo-Huber 的 delta，作用于对数空间
    "tc_weight_cap": 20.0,       # 低估权重的数值上限，防止极端倍率导致梯度爆炸
    # 折外验证（时间序列前向链 K 折）
    "tc_oof_folds": 5,
    "tc_oof_epochs": 50,         # 折外每折训练的轮数（比全量训练少，控制成本）
    # 输出命名：文件前缀，与 LGB 的 timeout_*/val_*/ooc_* 区分
    "tc_output_prefix": "tcnn_",
    "tc_pred_filename": "TransformerCNN预测数据.csv",

    # 图表设置
    "figure_size": (12, 6), "figure_dpi": 200,
    
    # 输出路径
    "model_dir": Path(__file__).parent.parent / "model",
    "asset_dir": Path(__file__).parent.parent / "assets",
    'report_dir': Path(__file__).parent.parent / "report",
    'output_dir': Path(__file__).parent.parent / "output",
    'base_dir':  Path(__file__).parent.parent

    
}

