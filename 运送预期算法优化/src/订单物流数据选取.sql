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
payment_type
FROM
order_payments op
WHERE payment_sequential = 1
),

pr AS (

SELECT 
order_id,
ROUND(SUM(product_weight_g)/1000,3) '商品总质量_kg',
ROUND(SUM(product_length_cm * product_height_cm * product_width_cm)/1000,3) '商品总体积_dm3'
FROM order_items oi
LEFT JOIN products p ON oi.product_id=p.product_id
GROUP BY order_id
),

t AS
(
SELECT
order_purchase_timestamp '下单时间',
oi.seller_id,
c.customer_unique_id,
IFNULL(product_category_name,'None') '商品类型',
freight_value '运费',
oi.price '价格',
op.payment_type '付款方式',
商品总质量_kg,
商品总体积_dm3,
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

order_approved_at '审核通过时间',
shipping_limit_date '预计最晚发货时间',
order_estimated_delivery_date '预计送达时间',
order_delivered_carrier_date '实际发货时间',
order_delivered_customer_date '实际送达时间',
CAST(strftime('%H', order_purchase_timestamp) AS INTEGER) '下单时刻',
CAST(strftime('%H', order_delivered_carrier_date) AS INTEGER) '发货时刻',
CAST(strftime('%H', order_approved_at) AS INTEGER) '审核通过时刻',
CAST(strftime('%H', order_delivered_customer_date) AS INTEGER) '收货时刻',
ROUND(julianday(order_approved_at) - julianday(order_purchase_timestamp) ,2) '审核周期',
ROUND(julianday(order_delivered_carrier_date) - julianday(order_approved_at) ,2) '发货周期',
ROUND(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) ,2) '送达时间差',
ROUND(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp) ,2) '总运送周期',
ROUND(julianday(o.order_estimated_delivery_date) - julianday(o.order_purchase_timestamp) ,2) '预期总周期',

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
LEFT JOIN op
ON op.order_id = oi.order_id
LEFT JOIN pr 
ON pr.order_id = oi.order_id


WHERE order_status = 'delivered'
AND order_purchase_timestamp >= '2017-08-01'
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

ORDER BY 下单时间)

SELECT 
*
FROM t














