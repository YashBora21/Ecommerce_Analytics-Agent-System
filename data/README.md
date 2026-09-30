# Ecommerce dataset

`ecommerce_sales_analytics_5000.csv` is a deterministic 5,000-order sample derived from the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

The source contains anonymized commercial orders from Brazil between 2016 and 2018 and is licensed under CC BY-NC-SA 4.0. This derived dataset is intended for non-commercial use under the same license.

## Sampling and grain

- One row represents one order.
- Orders are selected by sorting the SHA-256 hash of `order_id` and taking the first 5,000, producing a stable sample across the full source date range.
- `quantity` counts order-item rows; Olist has no separate unit-quantity field.
- Item, freight, and payment values are summed per order.
- `order_value` equals `item_value + freight_value`.
- For orders containing multiple categories or sellers, `product_category` and `seller_id` identify the contributor with the highest item value.
- For split payments, `payment_method` and `payment_installments` come from the largest payment; `payment_value` includes all payments.
- `review_score` is the mean when an order has multiple review records.
- `delivery_days` is elapsed time from purchase to customer delivery, expressed in days.

## Missing-value conventions

The CSV has no empty cells. Missing source facts use explicit values instead of invented business data:

- `not_available` means a canceled or unavailable order has no product or seller record.
- `review_score = 0` means no review was submitted; rating calculations must use scores from 1 through 5.
- `delivered_date = not_delivered` and `delivery_days = -1` mean the order was not delivered; delivery calculations must use `delivery_days >= 0`.
- Customer and seller state abbreviations are expanded to full Brazilian state names.

