-- The 10 business questions from the baseline report, plus one query for the
-- new K-Means segments. Run against the `customer` table.

-- Q1. Total revenue by gender
SELECT gender, SUM(purchase_amount) AS revenue
FROM customer
GROUP BY gender
ORDER BY revenue DESC;

-- Q2. Customers who used a discount but still spent more than the average purchase
SELECT customer_id, purchase_amount
FROM customer
WHERE discount_applied = 'Yes'
  AND purchase_amount >= (SELECT AVG(purchase_amount) FROM customer)
ORDER BY purchase_amount DESC;

-- Q3. Top 5 products by average review rating
SELECT item_purchased, ROUND(AVG(review_rating), 2) AS avg_rating
FROM customer
GROUP BY item_purchased
ORDER BY AVG(review_rating) DESC
LIMIT 5;

-- Q4. Average purchase amount: Standard vs Express shipping
SELECT shipping_type, ROUND(AVG(purchase_amount), 2) AS avg_purchase
FROM customer
WHERE shipping_type IN ('Standard', 'Express')
GROUP BY shipping_type;

-- Q5. Subscribers vs non-subscribers: customers, average spend, total revenue
SELECT subscription_status,
       COUNT(*)                         AS customers,
       ROUND(AVG(purchase_amount), 2)   AS avg_spend,
       SUM(purchase_amount)             AS total_revenue
FROM customer
GROUP BY subscription_status
ORDER BY total_revenue DESC;

-- Q6. Top 5 products with the highest share of discounted purchases
SELECT item_purchased,
       ROUND(100.0 * SUM(CASE WHEN discount_applied = 'Yes' THEN 1 ELSE 0 END)
             / COUNT(*), 2) AS discount_rate_pct
FROM customer
GROUP BY item_purchased
ORDER BY discount_rate_pct DESC
LIMIT 5;

-- Q7. Rule-based segmentation by previous purchases (the hand-written rule
--     that the K-Means pipeline replaces). Thresholds: 1 = New,
--     2-10 = Returning, more than 10 = Loyal.
WITH customer_type AS (
    SELECT customer_id,
           CASE WHEN previous_purchases = 1 THEN 'New'
                WHEN previous_purchases BETWEEN 2 AND 10 THEN 'Returning'
                ELSE 'Loyal'
           END AS customer_segment
    FROM customer
)
SELECT customer_segment, COUNT(*) AS customers
FROM customer_type
GROUP BY customer_segment
ORDER BY customers DESC;

-- Q8. Top 3 most purchased products within each category
WITH item_counts AS (
    SELECT category, item_purchased, COUNT(*) AS total_orders,
           ROW_NUMBER() OVER (PARTITION BY category
                              ORDER BY COUNT(*) DESC, item_purchased) AS item_rank
    FROM customer
    GROUP BY category, item_purchased
)
SELECT item_rank, category, item_purchased, total_orders
FROM item_counts
WHERE item_rank <= 3
ORDER BY category, item_rank;

-- Q9. Are repeat buyers (more than 5 previous purchases) more likely to subscribe?
SELECT subscription_status, COUNT(*) AS repeat_buyers
FROM customer
WHERE previous_purchases > 5
GROUP BY subscription_status;

-- Q10. Revenue by age group
SELECT age_group, SUM(purchase_amount) AS revenue
FROM customer
GROUP BY age_group
ORDER BY revenue DESC;

-- Q11 (new). K-Means segment summary
SELECT segment,
       COUNT(*)                                          AS customers,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_base,
       ROUND(AVG(purchase_rate_per_year), 1)             AS avg_purchases_per_year,
       ROUND(AVG(purchase_amount), 2)                    AS avg_spend,
       ROUND(100.0 * AVG(discount_flag), 1)              AS discount_usage_pct,
       SUM(purchase_amount)                              AS revenue
FROM customer
GROUP BY segment
ORDER BY customers DESC;
