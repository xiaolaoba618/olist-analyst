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

temp AS
(
SELECT
order_purchase_timestamp '下单时间',
oi.seller_id,
c.customer_unique_id,
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

ROUND(julianday(order_approved_at) - julianday(order_purchase_timestamp) ,2) '审核时间',
ROUND(julianday(shipping_limit_date) - julianday(order_approved_at),2) '预计最长发货周期',
ROUND(julianday(order_estimated_delivery_date) - julianday(order_purchase_timestamp) ,2) '预计总运送周期',
ROUND(julianday(order_delivered_carrier_date) - julianday(order_approved_at) ,2) '实际发货时长',
ROUND(julianday(order_delivered_customer_date) - julianday(order_delivered_carrier_date) ,2) '实际物流时长',
ROUND(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp) ,2) '实际总运送周期',
CASE WHEN ROUND(julianday(order_delivered_carrier_date) - julianday(shipping_limit_date) ,2) > 0 THEN
ROUND(julianday(order_delivered_carrier_date) - julianday(shipping_limit_date) ,2) ELSE 0 END  '发货时长延迟时间',
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


WHERE order_status = 'delivered'
AND order_purchase_timestamp >= '2017-09-01'
AND  order_delivered_customer_date IS NOT NULL --存在客户实际送达时间缺失
AND  order_delivered_carrier_date IS NOT NULL  --存在商家实际发货时间缺失
AND  order_approved_at IS NOT NULL            --存在审核时间缺失
AND  gc.geolocation_city  IS NOT NULL     --存在客户未知位置邮编编码
AND gs.geolocation_city  IS NOT NULL     --存在商家未知位置邮编编码
AND ROUND(julianday(shipping_limit_date) - julianday(order_approved_at),2) >= 0 --存在预估发货时间早于审核时间
AND ROUND(julianday(order_delivered_carrier_date) - julianday(order_approved_at) ,2) >= 0 --存在实际发货时间早于审核时间
AND ROUND(julianday(order_delivered_customer_date) - julianday(order_delivered_carrier_date) ,2) >= 0 --存在实际运送达到时间早于实际发货时间
AND ROUND(julianday(shipping_limit_date) - julianday(order_approved_at) ,2) IS NOT NULL
AND ROUND(julianday(order_delivered_carrier_date) - julianday(order_approved_at) ,2) IS NOT NULL
AND ROUND(julianday(order_delivered_customer_date) - julianday(order_delivered_carrier_date) ,2)  IS NOT NULL
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
ORDER BY 下单时间)

SELECT *
FROM temp 
WHERE
实际最终延迟时间 > 0
AND 
strftime('%Y-%m-%d', `下单时间`) >= '2018-02-20'
AND
strftime('%Y-%m-%d', `下单时间`) <= '2018-03-21';













