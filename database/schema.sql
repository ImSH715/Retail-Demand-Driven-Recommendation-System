CREATE TABLE IF NOT EXISTS departments (
    department_id INTEGER PRIMARY KEY,
    department_name VARCHAR(100) NOT NULL
);
CREATE TABLE IF NOT EXISTS aisles (
    aisle_id INTEGER PRIMARY KEY,
    aisle_name VARCHAR(100) NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    aisle_id INTEGER REFERENCES aisles(aisle_id),
    department_id INTEGER REFERENCES departments(department_id)
);
CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    eval_set VARCHAR(20) NOT NULL,
    order_number INTEGER NOT NULL,
    order_dow INTEGER,
    order_hour_of_day INTEGER,
    days_since_prior_order INTEGER
);
CREATE TABLE IF NOT EXISTS order_products (
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    add_to_cart_order INTEGER,
    reordered INTEGER CHECK (reordered IN (0, 1)),
    PRIMARY KEY (order_id, product_id)
);
CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_order_products_product ON order_products(product_id);
