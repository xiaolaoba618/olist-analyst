SELECT 
    order_id,
    order_status,
    order_purchase_timestamp AS 下单时间,
    order_approved_at AS 审核时间,
    order_delivered_customer_date AS 送达时间,
    -- 计算一下审核比送达晚了几天
    ROUND(julianday(order_approved_at) - julianday(order_delivered_customer_date), 2) AS 审核晚于送达天数
FROM 
    orders
WHERE 
    -- 核心条件：审核时间 > 送达时间
    julianday(order_approved_at) > julianday(order_delivered_customer_date)
ORDER BY 
    审核晚于送达天数 DESC;