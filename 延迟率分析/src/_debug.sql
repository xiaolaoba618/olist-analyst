SELECT 
order_id,
SUM(product_weight_g),
SUM(product_length_cm * product_height_cm * product_width_cm)
FROM order_items oi
LEFT JOIN products p ON oi.product_id=p.product_id
GROUP BY order_id

