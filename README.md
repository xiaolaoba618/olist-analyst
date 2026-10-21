# 巴西电商 Olist 订单数据分析

基于 Kaggle Olist 公开数据集（2016-09 ~ 2018-08，约 10 万笔订单），围绕平台真实经营痛点做归因分析与方案测算。

## 一、要解决的问题

按业务紧迫度排了两条主线：

| 主线 | 业务痛点 | 要回答的问题 |
|---|---|---|
| **履约延迟** | 延迟订单评分 **2.57 星** vs 准时 **4.29 星**，1 星差评占 11.5% | 延迟是谁的责任？砍卖家、改线路还是调时效承诺？ |
| **客户复购** | 96,096 个客户里复购率仅 **3.12%** | 谁有救、什么时候召回、首单这笔账划不划算 |

两条线最终都落到可执行动作：卖家分级名单、线路延时配置表、ETA 调整方案的成本收益账。

## 二、技术流程

每个问题都走同一条三段式流水线，各段各司其职：

```
SQLite 取数构造变量  →  R 清洗与统计检验  →  Python 建模输出结果
```

**① SQL：取数 + 构造变量**
直接从 `db/olist.db` 写 SQL 完成多表关联和特征派生，而不是导出后再拼。典型操作：用 `geolocation` 经纬度算客户-卖家球面距离、用五个时间戳算审批/发货/运送各段时长、按 `order_id` 聚合避免一对多放大行数。

**② R：数据清洗 + 统计验证**
负责判断"这个因子到底有没有用"。用 `case_when` 构造归因标签（如把每单拆成纯发货/纯物流/共责/潜在发货风险四类），再用卡方检验、Wilcoxon、GLM 优势比模型验证显著性，通过的因子才进入下一步。

**③ Python：建模 + 输出**
做规则分级、优化求解和蒙特卡洛模拟，产出可直接落地的名单与配置表。

**贯穿全程的可视化挑因子**：`*_data.py` 系列脚本按州、品类、时间峰值、评分区间等维度分别聚合，喂给 `report.html` / `dashboard.html` 可视化看板。先看图发现异常（比如 2018 年 2-3 月延迟率突增到 21%），再回头用 R 检验、用 Python 构造特征——**看图提假设、统计做验证、建模出结果**。

## 三、目录结构

```
create_db.py                    CSV → SQLite 建库（输出到 db/olist.db）
db/olist.db                     SQLite 数据库，9 张表
data/raw/                       原始 CSV
数据字典.html                    表结构、字段释义、关联关系

延迟率分析/                      主线一：履约延迟
  src/快递过程分析.sql            取数：距离、时长、归因变量
  src/data_processed.r           R：归因标签构造 + 显著性检验
  src/identify_express_risks_data.py   线路风险识别与延时方案求解
  src/precise_gmv_simulated.py   ETA 调整三方案蒙特卡洛
  初步报告.md                     完整结论报告
  report.html / dashboard.html   可视化看板

客户分析以及复购率判断/            主线二：客户复购
  src/客户留存数据选取.sql         取数：客户级订单宽表
  src/data_processed.R           R：清洗与复购特征构造
  src/time_series_repurchase.py  复购间隔分布
  src/customer_eval.py           客单分层与高价值名单
```

## 四、复现

```bash
# 1. 建库（可重复运行，会重建）
D:\anaconda3\envs\DA_TOOL\python.exe create_db.py

# 2. 取数：用 SQLite 客户端执行各 src/*.sql

# 3. 清洗与检验：R 运行 src/data_processed.r / .R

# 4. 建模输出
D:\anaconda3\envs\DA_TOOL\python.exe 延迟率分析/src/identify_express_risks_data.py
D:\anaconda3\envs\DA_TOOL\python.exe 延迟率分析/src/precise_gmv_simulated.py
```

## 五、口径约定

所有分析统一遵守，避免数字对不上：

- **客户主体**用 `customer_unique_id`（96,096），不用订单级的 `customer_id`（99,441）
- **GMV** = `SUM(price + freight_value)`，且只取 `order_status = 'delivered'`
- **有效窗口**截至 2018-08-31（9 月后仅 20 笔未签收尾单，纳入会造成趋势断崖）
- **延迟定义**：实际签收时间 > `order_estimated_delivery_date`
- **归因标签**分四类：纯发货 / 纯物流 / 共责 / 发货延迟但未超时

## 六、主要结论

**履约延迟**：订单级归因显示纯物流仅占 6.9%，卖家发货责任合计 9.1%。A 类高风险卖家 117 家（GMV 敞口 53.6 万），但只贡献 5.8% 的超时单——杠杆主要在线路与时效预测，不在清退卖家。ETA 调整方案在订单级口径下单救成本 27 元/单，线路级口径下 286 元/单。

**客户复购**：复购率 3.12% 这一数字被观察期稀释，按 cohort 校正后观察满 6 个月的客户复购率约 3.7%；且 34% 的"复购"实为同日拆单。首单评分与复购呈非单调关系，客单价越高复购率反而越低（<100 元 3.99% → >1000 元 1.90%）。
