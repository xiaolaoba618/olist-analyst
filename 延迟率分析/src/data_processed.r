library(tidyverse)
df <- read.csv("延迟率分析/data/订单时间段分析.csv",fileEncoding = 'UTF-8')




library(lubridate)
library(dplyr)

df <- df %>%
  mutate(
    超时因素 = case_when(
      # ================= 发货延迟时间 > 0 的情况 =================
      
      # 1. 实际最终未延迟（<=0），但发货延迟了，属于潜在风险
      发货时长延迟时间 > 0 & 实际最终延迟时间 <= 0 ~ "发货延迟（风险）",
      
      # 2. 最终延迟了，且延迟时间小于等于发货延迟的时间（说明全是发货的锅）
      发货时长延迟时间 > 0 & 实际最终延迟时间 > 0 & 实际最终延迟时间 <= 发货时长延迟时间 ~ "发货延迟导致延期",
      
      # 3. 最终延迟了，且延迟时间大于发货延迟的时间（说明发货和物流都有锅）
      发货时长延迟时间 > 0 & 实际最终延迟时间 > 发货时长延迟时间 ~ "发货/运送均延期",
      
      
      # ================= 发货延迟时间 <= 0 的情况 =================
      
      # 4. 发货没延迟，但最终延迟了（物流的锅）
      发货时长延迟时间 <= 0 & 实际最终延迟时间 > 0 ~ "运送延迟导致延期",
      
      # 5. 发货没延迟，最终也没延迟
      发货时长延迟时间 <= 0 & 实际最终延迟时间 <= 0 ~ "未超时",
      
      # 兜底：如果数据中有 NA (空值)，防止报错
      TRUE ~ NA_character_ 
    )
  )



# 加载必要的包
library(dplyr)
library(stringr)
library(stringi)

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

# 查看结果
print(df)

write_csv(df,"延迟率分析/data/订单时间段分析清洗后.csv")


library(dplyr)
library(MASS)

### ================== 0. 补全被删的变量 ================== ###
# 重新生成 0/1 变量：实际最终延迟时间 > 0 为 1（超时），否则为 0（未超时）
df <- df %>%
  mutate(
    最终是否超时 = if_else(实际最终延迟时间 > 0, 1, 0)
  )


### ================== 1. 原有统计检验修复与保留 ================== ###

# 确保评分为有序因子 (1 < 2 < 3 < 4 < 5)
df$订单评分 <- factor(df$订单评分, levels = c(1, 2, 3, 4, 5), ordered = TRUE)

# 构建有序逻辑回归模型 (检验评分与延迟时间、运送周期的关系)
model_ord <- polr(订单评分 ~ 实际最终延迟时间 + 实际总运送周期, data = df, Hess = TRUE)
summary(model_ord)

# 计算 p-value
ctable <- coef(summary(model_ord))
p_values <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
(ctable <- cbind(ctable, "p value" = p_values)) 


library(brant)

# 执行 Brant 检验
brant_test <- brant(model_ord)
print(brant_test)

###拒绝原假设，不同分数段对评分影响不同，舍弃有序线性逻辑回归#####

#使用部分必留优势模型PPO
library(VGAM)
# 截断延迟时间（防止长尾极端值导致完全分离）
df_trimmed <- df %>%
  mutate(实际最终延迟时间 = pmin(实际最终延迟时间, 30)) # 超过30天的全算作30天

model_ppo_trimmed <- vglm(订单评分 ~ 实际最终延迟时间 + 实际总运送周期, 
                          data = df_trimmed, 
                          family = cumulative(parallel = FALSE ~ 实际最终延迟时间))
summary(model_ppo_trimmed)



# 判断运送距离和是否超时具有显著关系 (二分类逻辑回归)
model_distance <- glm(最终是否超时 ~ 两地直线距离, data = df, family = binomial(link = "logit"))
summary(model_distance)

# 1）优势比OR + 95%CI
OR_ci <- cbind(
  OR = exp(coef(model_distance)),
  exp(confint.default(model_distance))
)
print(OR_ci)

# 2）模型整体显著性：似然比检验 LRT（最重要！）
# 对比当前模型 vs 空模型（仅截距）
lrt <- anova(model_distance, test = "Chisq")
print(lrt)

# 3）Pseudo R² 伪R方（McFadden）
library(pscl)
pR2(model_distance)



# 5）Hosmer-Lemeshow 拟合优度检验
library(ResourceSelection)
hl <- hoslem.test(model_distance$y, fitted(model_distance), g = 10)
hl

# 6）ROC + AUC（区分度，预测效果）
library(pROC)
prob <- predict(model_distance, type = "response")
roc_obj <- roc(model_distance$y, prob)
auc(roc_obj)
plot(roc_obj, main = "ROC曲线")


####区分度很小，关系微弱



### ================== 2. 卡方检验 ================== ###
# 为了防止 NA 值导致卡方检验报错，先过滤掉相关列的 NA
df_test <- df %>% filter(!is.na(最终是否超时) & !is.na(发货州) & !is.na(商品类型))

# 检验发货州是否影响最终超时
chisq_state <- suppressWarnings(chisq.test(table(df_test$发货州, df_test$最终是否超时)))
print(chisq_state)

# 检验商品类型是否影响最终超时
chisq_category <- suppressWarnings(chisq.test(table(df_test$商品类型, df_test$最终是否超时)))
print(chisq_category)

# 检验卖家 ID 是否影响最终超时
chisq_seller <- suppressWarnings(chisq.test(table(df_test$seller_id, df_test$最终是否超时)))
print(chisq_seller)


### ================== 3. 针对“超时因素”的显著性检验 ================== ###
# Kruskal-Wallis 检验：看看不同“超时因素”下的评分分布是否显著不同
df_valid <- df %>% filter(!is.na(订单评分) & !is.na(超时因素))

kw_test <- kruskal.test(订单评分 ~ 超时因素, data = df_valid)
print(kw_test)

# 如果显著，做事后两两比较 (Wilcoxon)
pairwise.wilcox.test(as.numeric(df_valid$订单评分), df_valid$超时因素, p.adjust.method = "BH")


### ================== 4. 进阶：看“超时因素”在各州的分布 ================== ###
# 看看 MA 州到底是因为卖家发货慢，还是物流慢
table_state_factor <- table(df$发货州, df$超时因素)
print(table_state_factor["MA", ]) 

# 计算 MA 州各类超时原因占该州总超时的比例
prop.table(table_state_factor["MA", ]) * 100


  

# ================== 统计检验结论汇总 ==================

# 1. 【关于评分影响（有序逻辑回归）】：买家“极度厌恶晚到”。
#    实际延迟时间对评分的负面杀伤力，是绝对运送周期的 2.4 倍。
#    延迟天数每增加 1 天，高分概率下降 11.5%；而总路程多 1 天仅下降 5%。

# 2. 【关于超时与距离的关系（二分类逻辑回归）】：物理距离与超时风险显著正相关。
#    但是关系极其微弱。

# 3. 【关于不同超时因素对评分的影响（Kruskal-Wallis + Wilcoxon）】：
#    不同超时原因导致的评分差异极其显著（KW检验 p < 2.2e-16）。
#    事后比较显示，“未超时”的评分显著高于任何包含延迟的分类。
#    更关键的是：即使最终都延期了，“发货延迟导致延期”与“运送延迟导致延期”的评分也存在显著差异（p=0.00019）。
#    结合业务，这证明了：因卖家拖延导致的晚到，比纯物流慢导致的晚到，更容易激怒买家，惩罚更深。

# 4. 【关于超时归因的卡方检验】：发货州、商品类型、卖家个体能力均对最终是否超时存在极其显著的影响（p值均极小）。
#    证实了超时是区域性物流基建、特定品类发货难度以及卖家履约水平共同作用的结果。

# 5. 【MA 州超时真相揭秘（交叉表分析）】：MA 州超时率极高（近24%），但拆解超时原因后惊奇地发现：
#    纯“运送延迟导致延期”仅占极少数（2.7%）。
#    而由“发货延迟”引发的超时（风险+导致延期+均延期）合计占比高达 32.6%！
#    结论：MA 州的超时并非因为偏远或物流基建差，而是该州商家的发货行为存在严重的系统性违规。




#####################GMV模型构建########################

df_copy <- df
library(lubridate)

# 1. 强制把原数据的时间列转为纯日期格式（使用 lubridate 更稳）
df$下单时间 <- as.Date(df$下单时间)

df_gmv <- df %>%
  # 移除缺失值
  filter(!is.na(预计总运送周期), !is.na(实际总运送周期), !is.na(订单评分), !is.na(下单时间)) %>%
  
  mutate(
    # ================= 方案A：直接对原周期天数取整 =================
    # 把预计和实际的周期天数，四舍五入变成整数
    预计周期_天 = round(预计总运送周期, 0),
    实际周期_天 = round(实际总运送周期, 0),
    
    # 核心：直接用整数周期相减，绝对值是纯整数！
    延迟天数_天 = as.integer(实际周期_天 - 预计周期_天),
    
    # 提前时间 = 负数取反
    提前送达时间 = as.integer(pmax(0, -延迟天数_天)),
    
    # 送达状态
    送达状态 = case_when(
      延迟天数_天 > 0 ~ "延迟",
      延迟天数_天 < 0 ~ "提前",
      延迟天数_天 == 0 ~ "准时" 
    ),
    
    # 时间偏差率
    时间偏差率 = 延迟天数_天 / 预计总运送周期,
    
    # GMV特征
    总价格 = 价格 + 运费
  )
df_gmv <- df_gmv %>%
  mutate(
    延迟天数_天 = if_else(延迟天数_天 > 0,延迟天数_天,0)
    
  )


library(dplyr)
library(rstatix)

df<-df_copy
rm(df_copy)

# ================= 1. 描述性统计 =================
summary_stats <- df_gmv %>%
  group_by(送达状态) %>%
  summarise(
    订单数 = n(),
    平均评分 = mean(订单评分, na.rm = TRUE),
    五星好评率 = mean(订单评分 == 5, na.rm = TRUE) * 100
  )
print("==== 各送达状态评分概览 ====")
print(summary_stats)

# ================= 2. 整体 Kruskal-Wallis 检验 =================
kw_test <- kruskal.test(订单评分 ~ 送达状态, data = df_gmv)
print("\n==== Kruskal-Wallis 检验 ====")
print(kw_test)

# ================= 3. Dunn 事后两两比较 =================
dunn_res <- df_gmv %>% 
  dunn_test(订单评分 ~ 送达状态, p.adjust.method = "BH")
print("\n==== Dunn 两两比较 ====")
print(dunn_res)

# ================= 4. 分箱逻辑回归（寻找早到惩罚拐点） =================
# 4.1 构造分箱标签
df_gmv <- df_gmv %>%
  mutate(
    提前分箱 = case_when(
      送达状态 == "延迟" ~ "6_延迟",
      送达状态 == "准时" ~ "1_准时(基准)",
      `提前送达时间` == 1 ~ "2_提前1天",
      `提前送达时间` %in% c(2, 3) ~ "3_提前2-3天",
      `提前送达时间` >= 4 & `提前送达时间` <= 7 ~ "4_提前4-7天",
      `提前送达时间` > 7 ~ "5_过度提前(>7天)"
    )
  )

# 4.2 设定“准时”为基准组
df_gmv$提前分箱 <- relevel(as.factor(df_gmv$提前分箱), ref = "1_准时(基准)")

# 4.3 目标变量：是否为 5 星
df_gmv$是否5星 <- if_else(df_gmv$订单评分 == 5, 1, 0)

# 4.4 拟合回归模型（控制价格和实际周期，剥离其他因素）
model_early <- glm(是否5星 ~ 提前分箱 + 价格 + `实际周期_天`, 
                   data = df_gmv, family = binomial)

# 4.5 输出 OR 值（结果极其关键）
res_or <- exp(coef(model_early))
print("\n==== 提前/延迟天数对拿 5 星概率的 OR 值 ====")
print(res_or[grep("提前分箱", names(res_or))])


write_csv(df_gmv,"延迟率分析/processed_data/GMV分析.csv")


