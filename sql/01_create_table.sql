-- Customer table: one row per purchase, as written by `python -m src.train`
-- (artifacts/clustered_customers.csv = cleaned data + engineered features +
-- K-Means segment). Load it with `python -m src.load_to_postgres`.
DROP TABLE IF EXISTS customer;

CREATE TABLE customer (
    customer_id             INTEGER PRIMARY KEY,
    age                     INTEGER,
    gender                  TEXT,
    item_purchased          TEXT,
    category                TEXT,
    purchase_amount         NUMERIC(10, 2),
    location                TEXT,
    size                    TEXT,
    color                   TEXT,
    season                  TEXT,
    review_rating           NUMERIC(3, 1),
    subscription_status     TEXT,
    shipping_type           TEXT,
    discount_applied        TEXT,
    previous_purchases      INTEGER,
    payment_method          TEXT,
    frequency_of_purchases  TEXT,
    purchase_rate_per_year  NUMERIC(6, 1),
    purchase_frequency_days INTEGER,
    discount_flag           SMALLINT,
    age_group               TEXT,
    segment_id              INTEGER,
    segment                 TEXT
);
