SELECT


FROM order_items oi
LEFT JOIN 
products p 
ON p.product_id = oi.product_id
LEFT JOIN orders o ON 
o.order_id = oi.order_id
GROUP BY o.order_id
