# Olist ecommerce dataset

This directory contains the complete Brazilian E-Commerce Public Dataset by Olist, not the earlier 5,000-order derived sample.

## Source files

- `olist_orders_dataset.csv`: 99,441 orders
- `olist_customers_dataset.csv`: order-level customer records and stable customer IDs
- `olist_order_items_dataset.csv`: 112,650 item positions with product, seller, price, and freight
- `olist_order_payments_dataset.csv`: 103,886 payment transactions
- `olist_order_reviews_dataset.csv`: 99,224 review/order records
- `olist_products_dataset.csv`: 32,951 products
- `olist_sellers_dataset.csv`: 3,095 sellers
- `olist_geolocation_dataset.csv`: 1,000,163 zip-prefix coordinate observations
- `product_category_name_translation.csv`: Portuguese-to-English category names

## Important grains

Orders, items, payments, and reviews have different grains. Items, payments, and reviews can each contain multiple rows for one order. Aggregate each child table to one row per order before combining its values with another child table; otherwise joins can multiply rows and overstate totals.

`customer_id` identifies the customer record attached to one order, while `customer_unique_id` identifies a buyer across orders. Product sales use item `price`; freight is separate; paid value comes from `payment_value`.

Blank source fields are imported as SQL `NULL`. State values remain the original two-letter Brazilian codes. Source timestamps cover 2016-2018, so relative date questions must use the latest dataset timestamp rather than the current date.

Source: Brazilian E-Commerce Public Dataset by Olist. Follow the source license and terms for redistribution or commercial use.
