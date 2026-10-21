library(tidyverse)
library(dplyr)
library(stringr)
library(stringi)

df <- read.csv("客户分析以及复购率判断/data/订单时间段分析_用户画像.csv",fileEncoding = 'UTF-8')

df_c <- read.csv("客户分析以及复购率判断/processed_data/客户复购率分析.csv",fileEncoding = 'UTF-8')

df_cg <-  read.csv("客户分析以及复购率判断/processed_data/客户商品联合分析.csv",fileEncoding = 'UTF-8')

# 定义清洗函数
clean_pt_city <- function(x) {
  x <- stri_trans_general(x, "Latin-ASCII")  # 步骤1：去除音标符号 (ã->a, ç->c, é->e)
  x <- tolower(x)                            # 步骤2：统一转换为小写
  x <- gsub("[^a-z0-9]", " ", x)             # 步骤3：将非字母数字（如 -, ', .）替换为空格
  x <- str_squish(x)                         # 步骤4：去除首尾空格，并将中间连续的多个空格压缩为一个
  return(x)
}

# 应用到数据框（保留原列，新增 _清洗后 列）
df <- df %>%
  mutate(
    发货城市 = clean_pt_city(发货城市),
    收货城市 = clean_pt_city(收货城市)
  )



write_csv(df,"客户分析以及复购率判断/data/订单时间段分析_用户画像.csv")

df_all <-  read.csv("客户分析以及复购率判断/data/总订单.csv",fileEncoding = 'UTF-8')

df_all <- df_all %>%
  mutate(
    发货城市 = clean_pt_city(发货城市),
    收货城市 = clean_pt_city(收货城市)
  )

write_csv(df_all,"客户分析以及复购率判断/data/总订单.csv")

rm(df_all)

#用户复购率分层
df_c <- df_c %>%
  mutate(
    order_level = case_when(
      总订单数 == 1 ~ 1,
      总订单数 == 2 ~ 2,
      总订单数 >= 3 ~ 3
    ),
   超时率 =  round(超时订单数 / 总订单数 * 100,2)
  )

write_csv(df_c,"客户分析以及复购率判断/processed_data/客户复购率分析.csv")

# Kruskal-Wallis：三个等级的平均评分分布是否不同
kruskal.test(平均评分 ~ factor(order_level), data = df_c)
# 事后两两比较
library(ggpubr)
pairwise.wilcox.test(df_c$平均评分, factor(df_c$order_level), p.adjust.method = "bonferroni")


# 选中你要检验的全部连续变量
var_list <- c("超时率", "平均总运送周期", "平均运输直线距离", "平均运费", "平均评分")

# 存结果
res_kw <- list()
res_pair <- list()

for (v in var_list) {
  cat("==================== 变量：", v, "====================\n")
  
  # Kruskal-Wallis
  kw <- kruskal.test(df_c[[v]] ~ factor(df_c$order_level), data = df_c)
  print(kw)
  res_kw[[v]] <- kw
  
  # 两两Wilcoxon，Bonferroni校正
  pw <- pairwise.wilcox.test(df_c[[v]], factor(df_c$order_level),
                             p.adjust.method = "bonferroni")
  print(pw)
  res_pair[[v]] <- pw
}


#######结论：一次性卖家的超时率分布与复购客户有显著区别#############
#单独分析不同类型卖家
library(dplyr)
library(readr)



save_dir <- "客户分析以及复购率判断/processed_data"





# 多次购买用户
df_rep <- df_c %>% filter(order_level >=2 )



# 保存
write_csv(df_rep, file.path(save_dir, "复购客户.csv"))








###对客户首单进行分析

df_first <-  read.csv("客户分析以及复购率判断/data/总订单.csv",fileEncoding = 'UTF-8')

df_first <- df_first %>%
  mutate(
    发货城市 = clean_pt_city(发货城市),
    收货城市 = clean_pt_city(收货城市)
  )


# 转为Date，自动丢掉时分秒
df_first <- df_first %>%
  mutate(下单时间 = as.Date(下单时间))


df_first <- df_first %>%
  arrange(customer_unique_id, 下单时间) %>%  # 先按客户、时间升序排
  group_by(customer_unique_id) %>%
  slice_head(n = 1) %>%  # 每组取第1行 = 首单
  ungroup() %>%
  arrange(下单时间) 



df_first <- df_first %>%
  left_join(
    df_c %>% select(customer_unique_id, order_level),
    by = "customer_unique_id"
  ) %>%
  mutate(
    order_level = ifelse(is.na(order_level), -1, order_level)
  )
#若order_level为-1，则证明该用户自从2017-9-1日后从未在平台上下单过，划分为无法挽回的流失用户




write.csv(df_first, file = "客户分析以及复购率判断/processed_data/客户首单数据分析.csv", row.names = FALSE, fileEncoding = "UTF-8")



#反常的直觉：通过对自从创立以来所有用户首单订单数据评估，我们惊讶的发现，复购的用户反而会有更高的延迟率和更低的评分。经分析后，这很有可能是幸存者偏差导致。



library(dplyr)
library(ggplot2)

# 描述性统计：中位数和四分位距
df_first %>%
  group_by(order_level) %>%
  summarise(
    延迟中位数 = median(实际最终延迟时间, na.rm = TRUE),
    延迟_IQR = IQR(实际最终延迟时间, na.rm = TRUE),
    评分中位数 = median(订单评分, na.rm = TRUE),
    评分_IQR = IQR(订单评分, na.rm = TRUE),
    N = n()
  )

# 箱线图对比
ggplot(df_first, aes(x = factor(order_level), y = 实际最终延迟时间)) + geom_boxplot() + theme_minimal()
ggplot(df_first, aes(x = factor(order_level), y = 订单评分)) + geom_boxplot() + theme_minimal()

# 首单延迟
kruskal.test(实际最终延迟时间 ~ order_level, data = df_first)

# 首单评分
kruskal.test(订单评分 ~ order_level, data = df_first)
library(rstatix)

# 延迟的两两比较
dunn_delay <- df_first %>% dunn_test(实际最终延迟时间 ~ order_level, p.adjust.method = "BH")
print(dunn_delay)

# 评分的两两比较
dunn_score <- df_first %>% dunn_test(订单评分 ~ order_level, p.adjust.method = "BH")
print(dunn_score)


library(Hmisc)
smean.cl.boot(df_first$订单评分[df_first$order_level == 3], B = 1000) 

#证实幸存者偏差猜想，#########实际上只有流失用户，即2017-9-1日后不再购买商品的用户与购买过商品的用户间具有显著差异。流失用户首单评分显著更低。########
#建立逻辑回归量化模型
library(broom)      # 用于整理模型结果
library(ggeffects)  # 用于画边际效应曲线



# --- 视角一：预测是否复购 ---
# 定义因变量：order_level 为 2 或 3 视为复购(1)，-1 或 1 视为未复购(0)
df_model <- df_first %>%
  filter(!is.na(实际最终延迟时间), !is.na(订单评分)) %>%
  mutate(
    是否复购 = if_else(order_level >= 2, 1, 0),
    # 视角二可用：是否流失 = if_else(order_level == -1, 0, 1)
  )

# 构建逻辑回归模型 (加入评分和延迟时间)
model_repurchase <- glm(是否复购 ~ 实际最终延迟时间 + 订单评分, 
                        data = df_model, 
                        family = binomial(link = "logit"))

# 查看模型结果
summary(model_repurchase)

# 计算 OR 值（优势比），极其重要！
# 解释：如果延迟的 OR 值小于 1，说明延迟越大，复购几率越低
exp(coef(model_repurchase))
##均不显著，需要抛弃该模型
rm(model_repurchase)
library(randomForest)


# 1. 数据预处理（解决 NA 问题）
df_model_rf <- df_model %>%
  # 选出你要用的变量
  select(是否复购, 实际最终延迟时间, 订单评分, 商品类型, 运费, 两地直线距离) %>%
  
  # 【关键修复】直接删除包含任何 NA 的行，并确保复购是因子类型
  na.omit() %>%
  
  # 【防坑修复】如果商品类型超过 53 个类别，R 的 RF 会报错。
  # 这里直接用字符型或者数值型替代（或者只用最常见的品类）
  mutate(
    是否复购 = as.factor(if_else(是否复购 == 1, "Yes", "No")),
    # 随机森林不支持超过 53 个类别的因子，我们把商品类型转为字符型，或者直接删掉这个变量
    # 如果一定要控制品类，建议把低频品类合并为 "Other"
    商品类型 = as.character(商品类型) 
  )

# 考虑到 R 原生随机森林的限制，如果带上"商品类型"依然报错（分类过多），
# 建议直接把这列删掉，直接跑下面的代码：
df_model_rf <- df_model_rf %>% select(-商品类型)

# 2. 处理极度不平衡的数据（复购率只有 2%）
# 随机森林如果不平衡，会全部预测为"No"。我们需要对样本进行欠采样平衡。
# 提取“是”和“否”的样本
rf_yes <- df_model_rf %>% filter(是否复购 == "Yes")
rf_no <- df_model_rf %>% filter(是否复购 == "No")

# 按照“Yes”的数量，从“No”中随机抽取等量样本
set.seed(123) # 保证结果可复现
rf_no_sampled <- rf_no %>% sample_n(nrow(rf_yes))

# 重新组合成平衡数据集
df_rf_balanced <- bind_rows(rf_yes, rf_no_sampled)

# 3. 运行随机森林模型
rf_model <- randomForest(
  是否复购 ~ ., 
  data = df_rf_balanced, 
  ntree = 500, 
  importance = TRUE
)

# 4. 查看结果与特征重要性
print(rf_model)

# 画出重要性图（延迟时间到底排第几？）
varImpPlot(rf_model, main = "复购预测特征重要性 (随机森林)", pch = 19, col = "blue")

#######结论表明，通过首单用户体验量化是否复购极其困难，因为仅下单一次的用户占据了几乎所有用户，这样大的基数会打平一切可能存在的关系。###
rm(rf_no,rf_no,rf_no_sampled,rf_yes,rf_model,df_model_rf,df_rf_balanced,df_model)
#对存在复购的用户喜好分析

# anti_join 会保留 df_cg 的所有列，只剔除 df_c 中匹配上且 order_level == 1 的行
df_cg_filtered <- df_cg %>%
  anti_join(
    df_c %>% filter(order_level == 1) %>% select(customer_unique_id), 
    by = "customer_unique_id"
  )

# 方法二：常规 left_join + filter + select（逻辑更直观，防止有些环境不支持 anti_join）
df_cg_filtered <- df_cg %>%
  left_join(df_c %>% select(customer_unique_id, order_level), by = "customer_unique_id") %>%
  # 保留没有匹配上（NA）或者 order_level 不等于 1 的
  filter(is.na(order_level) | order_level != 1) %>%
  # 删掉 join 过来的 order_level 列，只保留 df_cg 原本的信息
  select(-order_level)

cat("原始 df_cg 行数:", nrow(df_cg), "\n")
cat("剔除单次购买用户后行数:", nrow(df_cg_filtered), "\n")

write_csv(df_cg_filtered,"客户分析以及复购率判断/processed_data/复购客户商品联合数据.csv")

#分析客户复购时间段

