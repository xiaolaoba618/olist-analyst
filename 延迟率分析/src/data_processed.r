library(tidyverse)
df <- read.csv("延迟率分析/data/订单时间段分析.csv",fileEncoding = 'UTF-8')
#去除发货地收货地为空，预计时间为空的数据



library(lubridate)
df <- df %>%
  # 时间字段转为日期时间
  mutate(
    # 标记订单是否发货超时、是否送达超时
    发货是否超时 = if_else(发货时长延迟时间 > 0, 1, 0),
    最终是否超时 = if_else(实际最终延迟时间 > 0, 1, 0)
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


###统计检验部分#####

# 确保评分为有序因子 (1 < 2 < 3 < 4 < 5)
df$订单评分 <- factor(df$订单评分, levels = c(1, 2, 3, 4, 5), ordered = TRUE)

# 加载 MASS 包
library(MASS)

# 构建有序逻辑回归模型
model_ord <- polr(订单评分 ~ 实际最终延迟时间 + 实际总运送周期, data = df, Hess = TRUE)

# 查看摘要
summary(model_ord)

# 计算 p-value (为了精确输出)
ctable <- coef(summary(model_ord))
p_values <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
(ctable <- cbind(ctable, "p value" = p_values)) 
#####结果表明，最终延迟时间系数是总运动周期的两倍以上，证明相比于绝对路程，顾客极度厌恶延迟送达#####
# 判断运送距离和是否超时具有显著关系
# 构建二分类逻辑回归模型
model_distance <- glm(最终是否超时 ~ 两地直线距离, data = df, family = binomial(link = "logit"))

# 查看结果
summary(model_distance)

# 计算 OR值 (优势比)
exp(coef(model_distance))

#####每增加1km，订单超时几率增加0.04%，距离增加1000km，超时几率将激增至约50%

###判断发货州，商品类型与超时率的关系是否显著
# 构建交叉表并跑卡方检验
chisq_state <- chisq.test(table(df$发货州, df$最终是否超时))
print(chisq_state)

# 2. 检验商品类型是否影响最终超时
chisq_category <- chisq.test(table(df$商品类型, df$最终是否超时))
print(chisq_category)

chisq_seller <- chisq.test(table(df$seller_id, df$最终是否超时))
print(chisq_seller)
####卡方检验结果显著######


  
#统计学检验成果总结

#1. 关于评分影响（有序逻辑回归）：买家“极度厌恶晚到”。数据显示，实际延迟时间对评分的负面杀伤力，是绝对运送周期的 2.4 倍（延迟天数每增加1天，高分概率下降11.5%，而总路程多1天仅下降5%）。
#2. 关于超时与距离的关系（二分类逻辑回归）：物理距离与超时风险显著正相关。距离每增加 1000 公里，订单超时的几率将激增近 50%（OR=1.0004）。
#3. 关于超时归因（卡方检验）：发货州、商品类型以及卖家个体能力均对最终是否超时存在极其显著的影响（p值均极小）。这说明超时不仅是路程问题，更是区域性物流基建、特定品类发货难度以及卖家履约水平共同作用的结果。





