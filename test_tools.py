from tools.support_tools import (
    get_order_status,
    get_customer_details,
    create_support_ticket
)


print("\n========== ORDER TOOL ==========\n")

result = get_order_status(
    "TN1001"
)

print(result)


print("\n========== CUSTOMER TOOL ==========\n")

result = get_customer_details(
    "C001"
)

print(result)


print("\n========== TICKET TOOL ==========\n")

result = create_support_ticket(
    "C001",
    "Customer reports that the headphones are not working."
)

print(result)