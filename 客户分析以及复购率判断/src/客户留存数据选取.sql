WITH oi AS(
SELECT
order_id,
product_id,
seller_id,
shipping_limit_date,
SUM(price) price,
SUM(freight_value) freight_value
FROM order_items 
GROUP BY order_id
),
g AS (
SELECT 
geolocation_zip_code_prefix,
AVG(geolocation_lat) geolocation_lat,
AVG(geolocation_lng) geolocation_lng,
geolocation_city,
geolocation_state
FROM
geolocation
GROUP BY 
geolocation_zip_code_prefix
),
op AS(
SELECT
order_id,
count(*) '付款次数' ,
payment_installments '分期次数',
sum(payment_value) '总支付金额'
FROM 
order_payments
GROUP BY order_id
ORDER BY sum(payment_value) DESC


),
t AS(
SELECT
order_purchase_timestamp '下单时间',
oi.seller_id,
c.customer_unique_id,
o.order_status '订单状态',
IFNULL(product_category_name,'None') '商品类型',
freight_value '运费',
gs.geolocation_city AS '发货城市',
gs.geolocation_state AS '发货州',
gc.geolocation_city AS '收货城市',
gc.geolocation_state AS '收货州',
ROUND( 2 * 6371 * ATAN2(
     SQRT(
       POW(SIN(RADIANS((gs.geolocation_lat - gc.geolocation_lat)/2)), 2)
       + COS(RADIANS(gc.geolocation_lat)) * COS(RADIANS(gs.geolocation_lat))
       * POW(SIN(RADIANS((gs.geolocation_lng - gc.geolocation_lng)/2)), 2)
     ),
     SQRT(MAX(0,1 - (
       POW(SIN(RADIANS((gs.geolocation_lat - gc.geolocation_lat)/2)), 2)
       + COS(RADIANS(gc.geolocation_lat)) * COS(RADIANS(gs.geolocation_lat))
       * POW(SIN(RADIANS((gs.geolocation_lng - gc.geolocation_lng)/2)), 2)
     ))
   )),2) AS '两地直线距离',


ROUND(julianday(order_estimated_delivery_date) - julianday(order_purchase_timestamp) ,2) '预计总运送周期',
ROUND(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp) ,2) '实际总运送周期',
CASE WHEN ROUND(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) ,2 ) > 0 THEN
ROUND(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) ,2) ELSE 0 END 
 '实际最终延迟时间',
oi.price '价格',
IFNULL(rv.review_score,-1) '订单评分' --未评分为-1

 
-- order_purchase_timestamp,
-- order_approved_at,
-- shipping_limit_date,
-- order_estimated_delivery_date,
-- order_delivered_carrier_date,
-- order_delivered_customer_date,
-- gs.geolocation_zip_code_prefix ,
-- gs.geolocation_lat ,
-- gs.geolocation_lng ,
-- gc.geolocation_zip_code_prefix ,
-- gc.geolocation_lat ,
-- gc.geolocation_lng ,
-- c.customer_zip_code_prefix '用户邮编',
-- s.seller_zip_code_prefix '商家邮编'

FROM 
orders o
LEFT JOIN 
oi ON oi.order_id = o.order_id 
LEFT JOIN 
customers c ON o.customer_id = c.customer_id
LEFT JOIN 
sellers s ON oi.seller_id = s.seller_id
LEFT JOIN 
g gc ON gc.geolocation_zip_code_prefix = c.customer_zip_code_prefix
LEFT JOIN 
g gs ON gs.geolocation_zip_code_prefix = s.seller_zip_code_prefix
LEFT JOIN 
order_reviews rv 
ON rv.order_id = o .order_id 
LEFT JOIN products p
ON p.product_id = oi.product_id
LEFT JOIN  op
ON op.order_id = o.order_id 


WHERE  order_purchase_timestamp >= '2017-09-01'


-- AND s.seller_id ='2e90cb1677d35cfe24eef47d441b7c87' 
-- OR s.seller_id = '42bde9fef835393bb8a8849cb6b7f245'
-- OR s.seller_id = '5962468f885ea01a1b6a97a218797b0a'
-- OR s.seller_id = '870d0118f7a9d85960f29ad89d5d989a'
-- AND gc.geolocation_city IS NULL
-- AND ROUND( 2 * 6371 * ATAN2(
--      SQRT(
--        POW(SIN(RADIANS((gs.geolocation_lat - gc.geolocation_lat)/2)), 2)
--        + COS(RADIANS(gc.geolocation_lat)) * COS(RADIANS(gs.geolocation_lat))
--        * POW(SIN(RADIANS((gs.geolocation_lng - gc.geolocation_lng)/2)), 2)
--      ),
--      SQRT(MAX(0,1 - (
--        POW(SIN(RADIANS((gs.geolocation_lat - gc.geolocation_lat)/2)), 2)
--        + COS(RADIANS(gc.geolocation_lat)) * COS(RADIANS(gs.geolocation_lat))
--        * POW(SIN(RADIANS((gs.geolocation_lng - gc.geolocation_lng)/2)), 2)
--      ))
--    )),2) IS NULL

-- AND ROUND(julianday(shipping_limit_date) - julianday(order_approved_at) ,2) IS NULL
--AND julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) > 0
ORDER BY 下单时间),
c AS (
SELECT 
customer_unique_id,
COUNT(*) AS 总订单数,
SUM(CASE WHEN 实际最终延迟时间 > 0 THEN 1 ELSE 0 END) AS 超时订单数,
ROUND(AVG (价格),2) AS 平均价格,
ROUND(AVG(订单评分) ,2) AS 平均评分,
ROUND(AVG(运费) ,2) AS 平均运费,
ROUND(AVG(两地直线距离) ,2) AS 平均运输直线距离,
ROUND(AVG(预计总运送周期) ,2) AS 平均预测总周期,
ROUND(AVG(实际总运送周期) ,2) AS 平均总运送周期
FROM t 
GROUP BY customer_unique_id
),
goods AS(
SELECT 
商品类型,
ROUND(SUM(CASE WHEN 订单状态 = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),2) AS "订单完成率",
COUNT(DISTINCT seller_id) AS "售卖该类型的商家数",
COUNT(DISTINCT customer_unique_id) AS "购买过此产品的用户数",
(SELECT ROUND(SUM(CASE WHEN cnt >= 2 THEN 1 ELSE 0 END)*100.0 / COUNT(*),2)
 FROM (SELECT customer_unique_id, COUNT(*) cnt FROM t t2 WHERE t2.商品类型 = t.商品类型 AND t2.订单状态='delivered' GROUP BY customer_unique_id) sub) AS "该类型商品复购率"
FROM t
WHERE 商品类型 IS NOT NULL  
GROUP BY 商品类型
),
customer_goods AS (
SELECT
    customer_unique_id,
    商品类型,
    COUNT(*) AS 该客户此品类下单总次数,
    -- 复购次数：总下单次数 - 首单1单；买1次=0次复购
    CASE WHEN COUNT(*) >= 1 THEN COUNT(*) - 1 ELSE 0 END AS 该客户此品类复购次数,
    SUM(CASE WHEN 订单状态 = 'delivered' THEN 1 ELSE 0 END) AS 该客户此品类已送达订单数,
    SUM(CASE WHEN 实际最终延迟时间 > 0 THEN 1 ELSE 0 END) AS 该客户此品类超时订单数,
    ROUND(AVG(订单评分),2) AS 该客户此品类平均评分,
    ROUND(AVG(运费),2) AS 该客户此品类平均运费,
    ROUND(AVG(两地直线距离),2) AS 该客户此品类平均运输距离,
    ROUND(AVG(预计总运送周期),2) AS 该客户此品类平均预测周期,
    ROUND(AVG(实际总运送周期),2) AS 该客户此品类平均实际周期
FROM t
WHERE 商品类型 IS NOT NULL                --筛选无描述商品
GROUP BY customer_unique_id, 商品类型
)
SELECT * FROM c














