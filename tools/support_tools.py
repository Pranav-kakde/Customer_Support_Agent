"""
TechNova Customer Support Agent
Simulated E-Commerce Backend

This file contains demo data and business-action tools used by
the LangGraph customer-support agent.

IMPORTANT:
This is a simulated backend for the portfolio project.
It does not connect to a real e-commerce database.
"""

import uuid

from memory.customer_memory import (
    get_customer_memory,
    save_customer
)


# ============================================================
# CUSTOMER DATA
# ============================================================

CUSTOMERS = {
    "C001": {
        "customer_id": "C001",
        "name": "Rahul Sharma",
        "email": "rahul@example.com",
        "phone": "+91-9876543210",
    },
    "C002": {
        "customer_id": "C002",
        "name": "Priya Patil",
        "email": "priya@example.com",
        "phone": "+91-9876543211",
    },
}


# ============================================================
# ORDER DATA
# ============================================================

ORDERS = {
    "TN1001": {
        "order_id": "TN1001",
        "customer_id": "C001",
        "product": "TechNova Wireless Headphones",
        "status": "Shipped",
        "tracking_number": "TRK987654",
        "estimated_delivery": "2026-09-03",
        "payment_status": "Paid",
        "price": 2499,
        "return_eligible": True,
        "replacement_eligible": True,
        "cancellation_allowed": False,
        "refund_status": "Not Applicable",
    },
    "TN1002": {
        "order_id": "TN1002",
        "customer_id": "C002",
        "product": "TechNova Smart Watch",
        "status": "Delivered",
        "tracking_number": "TRK123456",
        "estimated_delivery": "2026-08-28",
        "payment_status": "Paid",
        "price": 3999,
        "return_eligible": True,
        "replacement_eligible": True,
        "cancellation_allowed": False,
        "refund_status": "Not Initiated",
    },
    "TN1003": {
        "order_id": "TN1003",
        "customer_id": "C001",
        "product": "TechNova Bluetooth Speaker",
        "status": "Processing",
        "tracking_number": None,
        "estimated_delivery": "2026-09-10",
        "payment_status": "Paid",
        "price": 1999,
        "return_eligible": False,
        "replacement_eligible": False,
        "cancellation_allowed": True,
        "refund_status": "Not Applicable",
    },
}


# ============================================================
# SUPPORT TICKETS
# ============================================================

SUPPORT_TICKETS = {}


# ============================================================
# HELPER
# ============================================================

def _generate_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:6].upper()}"


# ============================================================
# GET ORDER STATUS
# ============================================================

def get_order_status(order_id):
    """
    Get the current status and details of a TechNova order.
    """

    order_id = str(order_id).strip().upper()

    order = ORDERS.get(order_id)

    if not order:
        return {
            "success": False,
            "message": f"Order {order_id} was not found.",
        }

    return {
        "success": True,
        **order,
    }


# ============================================================
# GET CUSTOMER DETAILS
# ============================================================

def get_customer_details(customer_id):
    """
    Get TechNova customer details.
    """

    customer_id = str(customer_id).strip().upper()

    customer = CUSTOMERS.get(customer_id)

    if not customer:
        return {
            "success": False,
            "message": f"Customer {customer_id} was not found.",
        }

    return {
        "success": True,
        **customer,
    }


# ============================================================
# CREATE SUPPORT TICKET
# ============================================================

def create_support_ticket(customer_id, issue):
    """
    Create a simulated TechNova support ticket.
    """

    customer_id = str(customer_id).strip().upper()
    issue = str(issue).strip()

    if customer_id not in CUSTOMERS:
        return {
            "success": False,
            "message": f"Customer {customer_id} was not found.",
        }

    if not issue:
        return {
            "success": False,
            "message": "A support issue description is required.",
        }

    ticket_id = _generate_id("TN-TKT")

    ticket = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "issue": issue,
        "status": "Open",
    }

    SUPPORT_TICKETS[ticket_id] = ticket

    return {
        "success": True,
        **ticket,
        "message": f"Support ticket {ticket_id} has been created.",
    }


# ============================================================
# CANCEL ORDER
# ============================================================

def cancel_order(order_id):
    """
    Cancel an eligible TechNova order.

    Important:
    A repeated cancellation is treated as a failed operation,
    not a successful cancellation.
    """

    order_id = str(order_id).strip().upper()

    order = ORDERS.get(order_id)

    if not order:
        return {
            "success": False,
            "action": "cancel_order",
            "order_id": order_id,
            "message": f"Order {order_id} was not found.",
        }

    # Already cancelled
    if str(order.get("status", "")).lower() == "cancelled":
        return {
            "success": False,
            "action": "cancel_order",
            "order_id": order_id,
            "status": "Cancelled",
            "message": "This order has already been cancelled.",
        }

    if not order.get("cancellation_allowed", False):
        return {
            "success": False,
            "action": "cancel_order",
            "order_id": order_id,
            "status": order.get("status"),
            "message": "This order is no longer eligible for cancellation.",
        }

    order["status"] = "Cancelled"
    order["cancellation_allowed"] = False
    order["refund_status"] = "Pending"

    return {
        "success": True,
        "action": "cancel_order",
        "order_id": order_id,
        "status": "Cancelled",
        "message": f"Order {order_id} has been cancelled successfully.",
    }


# ============================================================
# REQUEST RETURN
# ============================================================

def request_return(order_id, reason):
    """
    Create a simulated return request for an eligible order.
    """

    order_id = str(order_id).strip().upper()
    reason = str(reason).strip()

    order = ORDERS.get(order_id)

    if not order:
        return {
            "success": False,
            "action": "request_return",
            "order_id": order_id,
            "message": f"Order {order_id} was not found.",
        }

    if not order.get("return_eligible", False):
        return {
            "success": False,
            "action": "request_return",
            "order_id": order_id,
            "message": "This order is not eligible for return.",
        }

    if str(order.get("status", "")).lower() != "delivered":
        return {
            "success": False,
            "action": "request_return",
            "order_id": order_id,
            "message": "A return can only be requested after the order is delivered.",
        }

    if not reason:
        return {
            "success": False,
            "action": "request_return",
            "order_id": order_id,
            "message": "A reason for the return is required.",
        }

    return_id = _generate_id("TN-RET")

    order["return_request_id"] = return_id
    order["return_reason"] = reason
    order["return_status"] = "Requested"

    return {
        "success": True,
        "action": "request_return",
        "order_id": order_id,
        "return_request_id": return_id,
        "status": "Requested",
        "message": f"Return request {return_id} has been created.",
    }


# ============================================================
# REQUEST REPLACEMENT
# ============================================================

def request_replacement(order_id, reason):
    """
    Create a simulated replacement request for an eligible order.
    """

    order_id = str(order_id).strip().upper()
    reason = str(reason).strip()

    order = ORDERS.get(order_id)

    if not order:
        return {
            "success": False,
            "action": "request_replacement",
            "order_id": order_id,
            "message": f"Order {order_id} was not found.",
        }

    if not order.get("replacement_eligible", False):
        return {
            "success": False,
            "action": "request_replacement",
            "order_id": order_id,
            "message": "This order is not eligible for replacement.",
        }

    if str(order.get("status", "")).lower() != "delivered":
        return {
            "success": False,
            "action": "request_replacement",
            "order_id": order_id,
            "message": "A replacement can only be requested after the order is delivered.",
        }

    if not reason:
        return {
            "success": False,
            "action": "request_replacement",
            "order_id": order_id,
            "message": "A reason for the replacement is required.",
        }

    replacement_id = _generate_id("TN-REP")

    order["replacement_request_id"] = replacement_id
    order["replacement_reason"] = reason
    order["replacement_status"] = "Requested"

    return {
        "success": True,
        "action": "request_replacement",
        "order_id": order_id,
        "replacement_request_id": replacement_id,
        "status": "Requested",
        "message": f"Replacement request {replacement_id} has been created.",
    }


# ============================================================
# CHECK REFUND STATUS
# ============================================================

def check_refund_status(order_id):
    """
    Check refund status for a TechNova order.
    """

    order_id = str(order_id).strip().upper()

    order = ORDERS.get(order_id)

    if not order:
        return {
            "success": False,
            "action": "check_refund_status",
            "order_id": order_id,
            "message": f"Order {order_id} was not found.",
        }

    return {
        "success": True,
        "action": "check_refund_status",
        "order_id": order_id,
        "refund_status": order.get(
            "refund_status",
            "Not Available",
        ),
        "message": (
            f"Refund status for {order_id}: "
            f"{order.get('refund_status', 'Not Available')}"
        ),
    }


# ============================================================
# GET SAVED CUSTOMER MEMORY
# ============================================================

def get_saved_customer_memory(customer_id):
    """
    Retrieve long-term saved memory for a TechNova customer.

    This memory contains durable information such as:
    - Communication preferences
    - Customer preferences
    - Useful support notes
    """

    customer_id = str(customer_id).strip().upper()

    # Validate customer
    if customer_id not in CUSTOMERS:
        return {
            "success": False,
            "message": f"Customer {customer_id} was not found.",
        }

    memory = get_customer_memory(customer_id)

    if not memory:
        return {
            "success": True,
            "customer_id": customer_id,
            "memory": None,
            "message": "No saved customer memory was found.",
        }

    return {
        "success": True,
        "customer_id": customer_id,
        "memory": memory,
        "message": "Saved customer memory retrieved successfully.",
    }


# ============================================================
# SAVE CUSTOMER MEMORY
# ============================================================

def save_customer_memory(
    customer_id,
    preferences=None,
    notes=None,
):
    """
    Save long-term customer preferences or notes.

    Only durable information should be saved.
    Temporary conversation details should remain in chat history.
    """

    customer_id = str(customer_id).strip().upper()

    # Validate customer
    if customer_id not in CUSTOMERS:
        return {
            "success": False,
            "message": f"Customer {customer_id} was not found.",
        }

    # Nothing to save
    if preferences is None and notes is None:
        return {
            "success": False,
            "message": "No customer memory information was provided.",
        }

    # Save to SQLite customer memory database
    save_customer(
        customer_id=customer_id,
        preferences=preferences,
        notes=notes,
    )

    return {
        "success": True,
        "customer_id": customer_id,
        "preferences": preferences,
        "notes": notes,
        "message": "Customer memory saved successfully.",
    }