<h1 align="center">
<strong><em style="font-family:'Brush Script MT', cursive;">ML Adventures</em></strong>
</h1>


## **Business Context**

GreenBasket Grocers is an online grocery delivery service headquartered in Austin, Texas, operating across 14 major metropolitan areas in the United States, including Austin, Dallas, Houston, Phoenix, Denver, Atlanta, Charlotte, Nashville, Tampa, Orlando, Columbus, Indianapolis, Kansas City, and Salt Lake City. The company runs a network of 141 dark stores supported by around 3,200 gig-based delivery partners, and fulfills close to 85,000 orders per day. It offers fresh produce, dairy, bakery, and packaged goods through scheduled 2-hour delivery windows, with an average order value of 42 dollars and a customer base of over 4 million registered users. GreenBasket has positioned itself as a convenient, reliable alternative to traditional grocery shopping, competing on delivery speed, product freshness, and slot availability.

A recurring operational challenge for the business is failed deliveries. On any given day, roughly 8 to 10 percent of orders either arrive late, get refused by the customer at the door, or come back undelivered due to reasons such as incorrect addresses, customers being unavailable, damaged perishables, or payment issues at the point of delivery. Each failed delivery costs the company between 12 and 28 dollars when accounting for wasted perishables, delivery partner payouts, reverse logistics, and customer service overhead. At the current failure rate, this translates to daily losses of nearly 150,000 to 200,000 dollars, in addition to the harder-to-measure damage to customer trust and repeat purchase behavior. Today, the operations team treats all orders the same way at dispatch, with no reliable method to identify which orders are at high risk of failing before the delivery is attempted.

If GreenBasket can accurately identify high-risk orders at the time of order placement, the operations team can take targeted actions such as confirmation calls, prepaid payment nudges, priority routing, or slot reassignment for those specific orders. Even a 25 percent reduction in the failure rate would save the company close to 18 million dollars annually, meaningfully cut down on perishable wastage, and improve the on-time delivery experience for millions of customers. It would also allow the company to allocate its customer service and dispatch resources more efficiently, focusing effort where it actually moves the needle rather than spreading it thinly across every order.


## **Objective**

To build a classification model that can predict whether an order is likely to fail delivery at the time of order placement, using features such as customer order history, past failure behavior, delivery address characteristics, order value, payment mode, item mix, chosen time slot, and delivery partner load, enabling the operations team to proactively intervene on high-risk orders through confirmation calls, prepaid, nudges, priority routing, or slot reassignment, and helping the company reduce perishable wastage, cut delivery costs, and improve overall customer experience.


## **Data Dictionary**

The dataset, `greenbasket_deliveries.csv`, contains 45,000 order-level records spanning a 45-day operational window across the 14 metropolitan markets served by GreenBasket.

### Identifiers & Metadata

| Column | Data Type | Description |
| --- | --- | --- |
| order_id | object | Unique identifier assigned to each order at the time of placement. |
| order_timestamp | datetime | Date and time when the order was placed. |
| customer_id | object | Unique identifier for the customer placing the order. |

### Customer History

| Column | Data Type | Description |
| --- | --- | --- |
| customer_tenure_days | int | Number of days since the customer registered on the platform. |
| past_order_count | int | Total number of prior orders placed by the customer before this one. |
| past_failure_count | int | Total number of prior orders that failed delivery for this customer. |
| past_failure_rate | float | Historical failure rate for the customer, smoothed with a Laplace prior. |
| avg_past_order_value_usd | float | Average value of the customer's prior orders in USD. |
| days_since_last_order | float | Number of days since the customer's most recent prior order. |

### Delivery Address

| Column | Data Type | Description |
| --- | --- | --- |
| delivery_city | object | City where the order is to be delivered. |
| delivery_pin_code | object | Postal code (ZIP) of the delivery location. |
| is_new_address | int | Flag indicating whether this is the first time delivering to this address for the customer. |
| address_type | object | Type of address (apartment, house, office, other). |
| floor_number | int | Floor number, applicable only for apartment deliveries. |
| distance_from_darkstore_km | float | Road distance from the fulfillment dark store to the delivery address. |
| pin_code_failure_rate | float | Historical failure rate at the delivery pin code, smoothed with a Laplace prior. |

### Order Details

| Column | Data Type | Description |
| --- | --- | --- |
| order_value_usd | float | Total value of the order in USD. |
| num_items | int | Total number of items in the order. |
| num_perishable_items | int | Number of perishable items in the order. |
| num_frozen_items | int | Number of frozen items in the order. |
| has_alcohol | int | Flag indicating whether the order contains alcohol, requiring age verification. |
| payment_mode | object | Payment method (prepaid_card, prepaid_wallet, cash_on_delivery). |
| discount_applied_pct | float | Percentage discount applied to the order. |

### Slot & Time

| Column | Data Type | Description |
| --- | --- | --- |
| delivery_slot | object | Chosen delivery window (morning, afternoon, evening, night). |
| day_of_week | object | Day of the week the order was placed. |
| is_weekend | int | Flag indicating whether the order was placed on a weekend. |
| lead_time_hours | float | Time in hours between order placement and the scheduled delivery slot. |

### Operational

| Column | Data Type | Description |
| --- | --- | --- |
| partner_current_load | float | Number of orders currently assigned to the delivery partner in the same slot. |
| darkstore_utilization_pct | float | Fulfillment capacity utilization of the originating dark store at the time of order. |
| weather_condition | object | Weather at the delivery location (clear, cloudy, rain, storm, snow). |
| temperature_f | float | Outside temperature at the delivery location in degrees Fahrenheit. |

### Target

| Column | Data Type | Description |
| --- | --- | --- |
| delivery_failed | int | Binary target variable indicating whether the delivery failed (1) or was successful (0). |PROBLEM STATEMENT

