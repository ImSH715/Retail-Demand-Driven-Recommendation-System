-- Business analytics using SQL window functions.
WITH product_sales AS (
    SELECT p.product_id, p.product_name, p.department_id,
           COUNT(*) AS purchase_count,
           SUM(op.reordered) AS reorder_count
    FROM order_products op
    JOIN products p ON p.product_id = op.product_id
    GROUP BY p.product_id, p.product_name, p.department_id
), ranked AS (
    SELECT *, RANK() OVER (PARTITION BY department_id ORDER BY purchase_count DESC) AS department_rank,
           ROUND(100.0 * reorder_count / NULLIF(purchase_count, 0), 2) AS reorder_rate
    FROM product_sales
)
SELECT * FROM ranked WHERE department_rank <= 10 ORDER BY department_id, department_rank;

-- Users' latest order and order interval.
SELECT user_id, order_id, order_number, days_since_prior_order,
       LAG(order_id) OVER (PARTITION BY user_id ORDER BY order_number) AS previous_order_id,
       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_number DESC) AS recency_rank
FROM orders;
