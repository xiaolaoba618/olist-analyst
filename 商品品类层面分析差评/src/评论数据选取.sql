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
)

SELECT 
review_id,
o.order_id,
oi.

MAX(CASE
  WHEN o.order_delivered_customer_date IS NOT NULL
   AND o.order_estimated_delivery_date IS NOT NULL
  THEN ROUND(
         julianday(date(o.order_delivered_customer_date))
       - julianday(date(o.order_estimated_delivery_date)),
         2)
  ELSE 0
END,0)
 '延迟天数'
FROM 
order_reviews re
LEFT JOIN 
orders o ON o.order_id = re.order_id 
LEFT JOIN 
oi ON o.order_id = oi.order_id 
LEFT JOIN 
products p ON p.product_id = 
GROUP BY 
re.review_id 