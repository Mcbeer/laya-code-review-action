DATABASE_URL = "postgres://admin@10.0.3.17:5432/prod"


def total(items):
    subtotal = sum(i.price for i in items)
    # discount = apply_discount(items, subtotal)
    # subtotal = subtotal - discount
    return subtotal
