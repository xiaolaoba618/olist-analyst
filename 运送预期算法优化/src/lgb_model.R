library(tidyverse)
library(dplyr)
library(stringr)
library(stringi)

df <- read.csv("运送预期算法优化/data/预测时间差.csv", fileEncoding = 'UTF-8')
df_processed <- read.csv("运送预期算法优化/data/订单数据_延迟优化后.csv", fileEncoding = 'UTF-8')

clean_pt_city <- function(x) {
  x <- stri_trans_general(x, "Latin-ASCII")  # 步骤1：去除音标符号 (ã->a, ç->c, é->e)
  x <- tolower(x)                            # 步骤2：统一转换为小写
  x <- gsub("[^a-z0-9]", " ", x)             # 步骤3：将非字母数字（如 -, ', .）替换为空格
  x <- str_squish(x)                         # 步骤4：去除首尾空格，并将中间连续的多个空格压缩为一个
  return(x)
}

# 应用到数据框（保留原列，新增 _清洗后 列）
df_processed <- df_processed %>%
  mutate(发货城市 = clean_pt_city(发货城市), 收货城市 = clean_pt_city(收货城市))

write.csv(df_processed, "运送预期算法优化/data/订单数据_延迟优化后.csv", row.names = FALSE, fileEncoding = "UTF-8")
summary(df)
var(df)

# 检验 df$`实际最终延迟时间` 分布特征
library(ggplot2)
library(moments)   # 偏度、峰度

x <- df$`实际最终延迟时间`
x_mean <- mean(x)
x_sd <- sd(x)

# ---------------------- 描述统计 + 偏度峰度 ----------------------
cat("========== 描述统计 ==========\n")
cat(sprintf("均值:%f  标准差:%f  最小值:%f  最大值:%f  中位数:%f  IQR:%f\n",
            x_mean, x_sd, min(x), max(x), median(x), IQR(x)))

sk <- skewness(x); ku <- kurtosis(x); ex_kurt <- ku - 3
cat("\n========== 偏度峰度 ==========\n")
cat(sprintf("偏度:%f  峰度:%f(正态=3)  超额峰度:%f(正态=0)\n", sk, ku, ex_kurt))

# ---------------------- 密度图 + 正态拟合曲线 ----------------------
p_density <- ggplot(df, aes(x = `实际最终延迟时间`)) +
  geom_histogram(aes(y = after_stat(density)), bins = 100, fill = "#3474b5", alpha = 0.4) +
  geom_density(color = "#3474b5", linewidth = 1) +
  stat_function(fun = dnorm, args = list(mean = x_mean, sd = x_sd),
                color = "red", linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = x_mean, color = "red", linestyle = "dashed") +
  labs(x = "实际最终延迟时间", y = "密度",
       title = "直方图+核密度+正态拟合曲线",
       subtitle = paste0("均值=", round(x_mean,4), " 标准差=", round(x_sd,4))) +
  theme_bw()
#print(p_density)
ggsave("运送预期算法优化/assets/img1.svg", plot = p_density, device = "svg", width = 8, height = 5, dpi = 300)

# ---------------------- 正态QQ图（核心看图判断厚尾/偏态） ----------------------
p_qq <- ggplot(df, aes(sample = `实际最终延迟时间`)) +
  stat_qq(color = "#3474b5", alpha = 0.2) +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(title = "正态QQ图", x = "理论正态分位数", y = "样本分位数") +
  theme_bw()
#print(p_qq)

# ---------------------- 应用延迟推迟后 -----------------------
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# ---------- 1. 合并成长表 ----------
comp <- bind_rows(
  data.frame(value = df$实际最终延迟,        group = "原始(实际最终延迟)"),
  data.frame(value = df_processed$送达时间差_优化后, group = "优化后(送达时间差_优化后)")
) %>% filter(!is.na(value))

# ---------- 2. 线性尺度密度对比（看整体形态） ----------
p_lin <- ggplot(comp, aes(x = value, fill = group, colour = group)) +
  geom_density(alpha = 0.35, linewidth = 0.9, adjust = 1.2) +
  scale_fill_manual(values = c("#E64B35", "#4DBBD5")) +
  scale_colour_manual(values = c("#E64B35", "#4DBBD5")) +
  labs(title = "延迟分布对比（线性尺度）", x = "延迟", y = "密度", fill = NULL, colour = NULL) +
  theme_bw(base_size = 13) + theme(legend.position = "top")
#p_lin

# ---------- 3. log1p 尺度密度对比（重点：看尾部是否变薄） ----------
p_log <- ggplot(comp, aes(x = value, fill = group, colour = group)) +
  geom_density(alpha = 0.35, linewidth = 0.9, adjust = 1.2) +
  scale_x_continuous(trans = scales::pseudo_log_trans(sigma = 1)) +
  scale_fill_manual(values = c("#E64B35", "#4DBBD5")) +
  scale_colour_manual(values = c("#E64B35", "#4DBBD5")) +
  labs(title = "延迟分布对比（pseudo-log 尺度，看厚尾）",
       x = "延迟 (pseudo-log)", y = "密度", fill = NULL, colour = NULL) +
  theme_bw(base_size = 13) + theme(legend.position = "top")
#p_log
ggsave("运送预期算法优化/assets/img2.svg", plot = p_log, device = "svg", width = 8, height = 5, dpi = 300)

# ---------- 4. ECDF 对比（尾部分位一目了然） ----------
p_ecdf <- ggplot(comp, aes(x = value, colour = group)) +
  stat_ecdf(linewidth = 1) +
  scale_x_continuous(trans = scales::pseudo_log_trans(sigma = 1)) +
  scale_colour_manual(values = c("#E64B35", "#4DBBD5")) +
  labs(title = "ECDF 对比（右尾越高越薄越好）",
       x = "延迟 (pseudo-log)", y = "累积概率", colour = NULL) +
  theme_bw(base_size = 13) + theme(legend.position = "top")
#p_ecdf

# ---------- 5. QQ 图对比（对同一分位网格） ----------
qs <- seq(0.01, 0.99, by = 0.01)
qq_df <- comp %>%
  group_by(group) %>%
  summarise(q = list(quantile(value, qs, na.rm = TRUE)), .groups = "drop") %>%
  tidyr::unnest(q) %>%
  mutate(p = rep(qs, times = n() / length(qs)))

p_qq <- ggplot(qq_df, aes(x = p, y = q, colour = group)) +
  geom_line(linewidth = 1) + geom_point(size = 1.2) +
  scale_y_continuous(trans = scales::pseudo_log_trans(sigma = 1)) +
  scale_colour_manual(values = c("#E64B35", "#4DBBD5")) +
  labs(title = "分位数曲线对比（同一分位下的值，越低说明尾部越轻）",
       x = "分位 p", y = "分位数值 (pseudo-log)", colour = NULL) +
  theme_bw(base_size = 13) + theme(legend.position = "top")
#p_qq

# ---------- 6. 关键尾部分位数值表（结论落地） ----------
tail_table <- comp %>%
  group_by(group) %>%
  summarise(
    n = n(), mean = mean(value), sd = sd(value),
    P50 = quantile(value, .50), P90 = quantile(value, .90), P95 = quantile(value, .95),
    P99 = quantile(value, .99), P999 = quantile(value, .999), Max = max(value),
    # 厚尾指标
    kurt = { x <- value; m <- mean(x); s <- sd(x); mean((x - m)^4) / s^4 - 3 },
    # 超过 P99 的占比
    over_p99 = mean(value > quantile(comp$value, .99)),
    .groups = "drop"
  ) %>%
  mutate(across(where(is.numeric), ~round(.x, 3)))
print(tail_table)

# ---------- 7. 峰值/尾部集中度：最极端的 1% 贡献占比 ----------
tail_ratio <- comp %>%
  group_by(group) %>%
  summarise(
    top1_share = sum(value[value >= quantile(value, .99)]) / sum(value),
    top5_share = sum(value[value >= quantile(value, .95)]) / sum(value),
    .groups = "drop"
  )
print(tail_ratio)

# 总运送周期
# ---------------------- 密度图 + 正态拟合曲线 ----------------------
x <- df_processed$`总运送周期`
log_x <- log(x)
mu_ln <- mean(log_x)
sigma_ln <- sd(log_x)

p_density_ln <- ggplot(df_processed, aes(x = `总运送周期`)) +
  geom_histogram(aes(y = after_stat(density)), bins = 100, fill = "#3474b5", alpha = 0.4) +
  geom_density(color = "#3474b5", linewidth = 1) +
  # 对数正态拟合曲线
  stat_function(fun = dlnorm, args = list(meanlog = mu_ln, sdlog = sigma_ln),
                color = "#e03c31", linewidth = 1, linetype = "dashed") +
  labs(x = "总运送周期", y = "密度",
       title = "直方图+核密度+对数正态拟合曲线",
       subtitle = paste0("meanlog=", round(mu_ln,4), "  sdlog=", round(sigma_ln,4))) +
  theme_bw()
#print(p_density_ln)
ggsave("运送预期算法优化/assets/img3.svg", plot = p_density_ln, device = "svg", width = 8, height = 5, dpi = 300)

# ----------------------
df <- df_processed
remove(comp,qq_df,tail_table,tail_ratio,p_lin,p_log,p_ecdf,p_qq,p_density_ln,x,log_x,mu_ln,sigma_ln,x_mean,x_sd,sk,ku,ex_kurt,p_density,qs,df_processed)
df_copy2 <- df

# ---------------------- LightGBM 模型数据准备 ----------------------
df <- df %>% select(-`订单评分`, -`送达时间差_优化后`,-'下单时刻',-'发货时刻',-'审核通过时刻',-'收货时刻', -'推荐延迟天数',-'审核通过时间',-'预计最晚发货时间',-'预计送达时间',-'实际发货时间',-'实际送达时间')
df_copy <- df
#summary(df)

# ---------------------- 特征构造----------------------
library(dplyr)
library(lubridate)
library(slider)

# 1. 基础特征计算 + 时间转换
df <- df %>%
  mutate(
    下单时间 = ymd_hms(下单时间), # 先转换为 POSIXct
    对数质量 = log(`商品总质量_kg` + 1),
    对数体积 = log(`商品总体积_dm3` + 1),
    对数价格 = log(`价格` + 1),
    对数运费 = log(`运费` + 1),
    对数距离 = log(`两地直线距离` + 1),
    是否超时 = if_else(送达时间差 > 0, 1, 0),
    运费占比 = round(if_else(`价格` == 0, NA_real_, `运费` / `价格`)*100, 2),
    单位距离运费 = round(`运费` / (`两地直线距离` + 1), 2),
    对数运送周期 = log(`总运送周期` + 1),
    星期_周一始 = (wday(下单时间) - 2) %% 7 + 1,
    是否周末 = ifelse(星期_周一始 %in% c(6,7), 1, 0)
  )

# 2. 计算滑动窗口特征
df <- df %>%
  mutate(
    平台_7天单量 = slide_index_dbl(下单时间, 下单时间, ~ sum(!is.na(.x)), .before = days(7)),
    平台_7天平均审核时长 = slide_index_dbl(审核周期, 下单时间, ~ mean(.x, na.rm = TRUE), .before = days(7))
  )

df <- df %>%
  arrange(seller_id, 下单时间) %>%
  group_by(seller_id) %>%
  mutate(
    卖家_7天单量 = slide_index_dbl(下单时间, 下单时间, ~ sum(!is.na(.x)), .before = days(7)),
    卖家_7天平均发货时长 = slide_index_dbl(发货周期, 下单时间, ~ mean(.x, na.rm = TRUE), .before = days(7)),
    卖家_7天超时率 = slide_index_dbl(是否超时, 下单时间, ~ mean(.x, na.rm = TRUE), .before = days(7))
  ) %>%
  ungroup()

# 3. 修复新卖家逻辑（关键：先判断，后填补）
df <- df %>%
  mutate(
    # 填补冷启动缺失值
    卖家_7天平均发货时长 = coalesce(卖家_7天平均发货时长, 平台_7天平均审核时长, 0),
    卖家_7天超时率 = coalesce(卖家_7天超时率, 0.05)
  )

# 4.卖家前七天单量以及单量的增幅二次信息
df <- df %>%
  arrange(seller_id, 下单时间) %>%
  group_by(seller_id) %>%
  mutate(
    # 用前7天（滞后7天）的单量作为基数
    卖家_前7天单量 = lag(卖家_7天单量, 7),
    # 计算增长率（防止除以0，基数+1）
    卖家_单量增幅 = (卖家_7天单量 - 卖家_前7天单量) / (卖家_前7天单量 + 1)
  ) %>%
  ungroup()

# 5. 卖家产能负荷率（当前单量 / 历史最高单量）
df <- df %>%
  arrange(seller_id, 下单时间) %>%
  group_by(seller_id) %>%
  mutate(
    # 累积历史最大7天单量（用 cummax）
    卖家_历史最高7天单量 = cummax(卖家_7天单量),
    # 当前负荷率 = 当前单量 / 历史最高单量
    卖家_产能负荷率 = 卖家_7天单量 / (卖家_历史最高7天单量 + 1)
  ) %>%
  ungroup()

# 6. 黑色星期五月
df <- df %>% mutate(是否黑五月份 = if_else(month(下单时间) == 11 | month(下单时间) == 12, 1, 0))

library(dplyr)
library(lubridate)

# 7.单独构造黑色星期五特征（黑五当天和黑五当周）
df <- df %>%
  mutate(
    # 提取年份和周数
    年份 = year(下单时间),
    # 用 epiweek 计算周数（这在流行病学和商业日历中非常标准）
    周数 = epiweek(下单时间),
    # 1. 构造“黑五当天”特征（2017年黑五是 11月24日，2018年是 11月23日）
    黑五日期_2017 = ymd("2017-11-24"),
    黑五日期_2018 = ymd("2018-11-23"),
    是否黑五当天 = if_else(
      as.Date(下单时间) == 黑五日期_2017 | as.Date(下单时间) == 黑五日期_2018, 1, 0
    ),
    # 2. 构造“黑五当周”特征 (包含黑五的那一周，即黑色星期五所在周)
    黑五周数_2017 = epiweek(黑五日期_2017),
    黑五周数_2018 = epiweek(黑五日期_2018),
    是否黑五当周 = if_else(
      (年份 == 2017 & 周数 == 黑五周数_2017) | (年份 == 2018 & 周数 == 黑五周数_2018), 1, 0
    )
  ) %>%
  # 清理中间变量
  select(-contains("黑五日期"), -contains("黑五周数"), -年份, -周数)

# 8. 构造大件特征（质量或体积超过 99% 分位数的订单）
weight_99 <- quantile(df$`商品总质量_kg`, 0.99, na.rm = TRUE)
volume_99 <- quantile(df$`商品总体积_dm3`, 0.99, na.rm = TRUE)

df <- df %>%
  mutate(是否为大件 = if_else(`商品总质量_kg` > weight_99 | `商品总体积_dm3` > volume_99, 1, 0))

library(dplyr)
library(lubridate)
library(slider)

# 8.构造动态的“历史路线特征”
# 1. 提取全网的全局先验（用于填补极冷门路线）
global_p90 <- quantile(df$`总运送周期`, 0.90, na.rm = TRUE)
global_late <- mean(df$`是否超时`, na.rm = TRUE)
m <- 10 # 平滑参数，单量少于 10 单的路线会强烈向全局均值靠拢

# 必须按路线和时间严格排序
df <- df %>% arrange(发货州, 收货州, 下单时间)

# 计算滑动历史特征
df <- df %>%
  group_by(发货州, 收货州) %>%
  mutate(
    # 截至当前订单之前，这条路线总共发了多少单
    cum_orders = row_number() - 1,
    # --- 总历史特征 ---
    # 使用 lag() 取上一单时的累计状态，完美规避当前单真实值的泄露
    hist_p90_total = lag(slide_index_dbl(`总运送周期`, 下单时间, ~ quantile(.x, 0.90, na.rm = TRUE, names = FALSE), .before = Inf)),
    hist_late_total = lag(slide_index_dbl(`是否超时`, 下单时间, ~ mean(.x, na.rm = TRUE), .before = Inf)),
    # --- 近期（过去 90 天）历史特征 ---
    hist_p90_90d = lag(slide_index_dbl(`总运送周期`, 下单时间, ~ quantile(.x, 0.90, na.rm = TRUE, names = FALSE), .before = days(90))),
    hist_late_90d = lag(slide_index_dbl(`是否超时`, 下单时间, ~ mean(.x, na.rm = TRUE), .before = days(90)))
  ) %>%
  ungroup() %>%
  mutate(
    # --- 贝叶斯平滑处理 ---
    路线P90_总历史 = (cum_orders * hist_p90_total + m * global_p90) / (cum_orders + m),
    路线超时率_总历史 = (cum_orders * hist_late_total + m * global_late) / (cum_orders + m),
    路线P90_近期90天 = (cum_orders * hist_p90_90d + m * global_p90) / (cum_orders + m),
    路线超时率_近期90天 = (cum_orders * hist_late_90d + m * global_late) / (cum_orders + m)
  ) %>%
  # 清理中间计算过程产生的列
  select(-cum_orders, -hist_p90_total, -hist_late_total, -hist_p90_90d, -hist_late_90d)

# 9.判定国际订单与 0/1 标签（根据你的新需求保留）
# 动态计算全网平滑后 P90 的 90% 分位数，作为偏远州的阈值
route_time_90 <- quantile(df$路线P90_总历史, 0.90, na.rm = TRUE)

df <- df %>%
  mutate(
    # 如果模型需要 0/1 标签辅助判断，可以保留。但强烈建议同时把“路线P90_总历史”等连续变量也喂给模型
    是否为偏远州 = if_else(
      路线P90_总历史 >= route_time_90 | (`两地直线距离` > 2500 ), 1, 0
    ),
    # 距离 > 6000km 视为国际/极长距离
    是否为国际订单 = if_else(`两地直线距离` > 6000, 1, 0)
  )

# 删除前 30 天数据与清理列
min_date <- min(df$下单时间, na.rm = TRUE)

df <- df %>%
  # 1. 删除前 30 天的预热数据
  filter(下单时间 >= min_date + days(30)) %>%
  # 2. 清理冗余列
  select(
    -`商品总质量_kg`, -`商品总体积_dm3`, -`价格`, -`运费`, -`两地直线距离`,
    -customer_unique_id, -`审核周期`, -`发货周期`, -`送达时间差`, -`是否超时`
  )

# 查看特征变量
colnames(df)

write.csv(df, "运送预期算法优化/processed_data/LightGBM.csv", row.names = FALSE, fileEncoding = "UTF-8") 





































