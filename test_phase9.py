"""
TechNova Customer Support Agent - Phase 9.4
Edge Cases + End-to-End Testing

Default:
    python test_phase9_4.py

Live Gemini tests:
    python test_phase9_4.py --live
"""

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(name, condition, details=""):
    global PASS, FAIL

    if condition:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name}")
        if details:
            print(f"       {details}")


def run_deterministic_tests():
    print("\n" + "=" * 64)
    print("PHASE 9.4 - DETERMINISTIC EDGE-CASE TESTS")
    print("=" * 64)

    try:
        from tools.support_tools import (
            get_order_status,
            get_customer_details,
            create_support_ticket,
            cancel_order,
            request_return,
            request_replacement,
            check_refund_status,
        )
    except Exception as exc:
        check("Import support tools", False, str(exc))
        return

    # ------------------------------------------------------------
    # 1. Valid order
    # ------------------------------------------------------------
    result = get_order_status("TN1001")
    check(
        "Valid order lookup",
        isinstance(result, dict)
        and result.get("order_id") == "TN1001",
        repr(result),
    )

    # ------------------------------------------------------------
    # 2. Invalid order
    # ------------------------------------------------------------
    result = get_order_status("TN9999")
    check(
        "Invalid order handled safely",
        isinstance(result, dict)
        and result.get("success") is False,
        repr(result),
    )

    # ------------------------------------------------------------
    # 3. Valid customer
    # ------------------------------------------------------------
    result = get_customer_details("C001")
    check(
        "Valid customer lookup",
        isinstance(result, dict)
        and result.get("customer_id") == "C001",
        repr(result),
    )

    # ------------------------------------------------------------
    # 4. Invalid customer
    # ------------------------------------------------------------
    result = get_customer_details("C9999")
    check(
        "Invalid customer handled safely",
        isinstance(result, dict)
        and result.get("success") is False,
        repr(result),
    )

    # ------------------------------------------------------------
    # 5. Eligible cancellation
    # ------------------------------------------------------------
    result = cancel_order("TN1003")
    check(
        "Eligible cancellation",
        isinstance(result, dict)
        and result.get("success") is True
        and result.get("status") == "Cancelled",
        repr(result),
    )

    # ------------------------------------------------------------
    # 6. Repeated cancellation
    # ------------------------------------------------------------
    result = cancel_order("TN1003")
    check(
        "Repeated cancellation rejected safely",
        isinstance(result, dict)
        and result.get("success") is False
        and result.get("status") == "Cancelled",
        repr(result),
    )

    # ------------------------------------------------------------
    # 7. Return for delivered order
    # ------------------------------------------------------------
    result = request_return(
        "TN1002",
        "Customer changed their mind"
    )
    check(
        "Eligible return request",
        isinstance(result, dict)
        and result.get("success") is True
        and str(result.get("status", "")).lower()
        in {"requested", "return requested"},
        repr(result),
    )

    # ------------------------------------------------------------
    # 8. Replacement for delivered order
    # ------------------------------------------------------------
    result = request_replacement(
        "TN1002",
        "Product is defective"
    )
    check(
        "Replacement request",
        isinstance(result, dict)
        and result.get("success") is True,
        repr(result),
    )

    # ------------------------------------------------------------
    # 9. Refund status lookup
    # ------------------------------------------------------------
    result = check_refund_status("TN1003")
    check(
        "Refund status lookup",
        isinstance(result, dict)
        and result.get("order_id") == "TN1003"
        and result.get("success") is True,
        repr(result),
    )

    # ------------------------------------------------------------
    # 10. Invalid refund lookup
    # ------------------------------------------------------------
    result = check_refund_status("TN9999")
    check(
        "Invalid refund order handled safely",
        isinstance(result, dict)
        and result.get("success") is False,
        repr(result),
    )

    # ------------------------------------------------------------
    # 11. Support ticket
    # ------------------------------------------------------------
    ticket = create_support_ticket(
        "C001",
        "Phase 9.4 test escalation"
    )
    check(
        "Support ticket creation",
        isinstance(ticket, dict)
        and ticket.get("success") is True
        and bool(ticket.get("ticket_id")),
        repr(ticket),
    )

    # ------------------------------------------------------------
    # 12. Memory directory
    # ------------------------------------------------------------
    db_dir = ROOT / "memory"
    check(
        "Memory directory exists",
        db_dir.exists(),
        str(db_dir),
    )

    print("\nDeterministic tests complete.")


def run_live_tests():
    print("\n" + "=" * 64)
    print("PHASE 9.4 - LIVE END-TO-END TESTS")
    print("=" * 64)
    print("These tests use Gemini and may consume API quota.\n")

    try:
        from langchain_core.messages import HumanMessage
        from graph.customer_support_graph import customer_support_graph
    except Exception as exc:
        check(
            "Import live customer-support graph",
            False,
            str(exc),
        )
        return

    scenarios = [
        (
            "Order tracking",
            "Where is my order TN1001?",
            ["TN1001", "shipped", "tracking"],
        ),
        (
            "Missing order ID clarification",
            "I want to return my order.",
            ["order", "id"],
        ),
        (
            "Return workflow",
            "Can I return TN1002?",
            ["TN1002", "return"],
        ),
        (
            "Product problem",
            "My headphones are not working. What should I do?",
            ["warranty", "troubleshoot", "headphones", "order"],
        ),
        (
            "Cancellation clarification",
            "I want to cancel my order.",
            ["order", "id"],
        ),
        (
            "Human support request",
            "I want to talk to customer support.",
            ["support", "ticket"],
        ),
    ]

    for name, prompt, expected_terms in scenarios:
        thread_id = f"phase9_4_{uuid.uuid4().hex[:10]}"
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        try:
            result = customer_support_graph.invoke(
                {
                    "messages": [
                        HumanMessage(content=prompt)
                    ]
                },
                config=config,
            )

            messages = result.get("messages", [])

            final_text = ""

            for message in reversed(messages):
                content = getattr(message, "content", "")

                if isinstance(content, str) and content.strip():
                    final_text = content.strip()
                    break

            lower = final_text.lower()

            matched = any(
                term.lower() in lower
                for term in expected_terms
            )

            check(
                name,
                bool(final_text) and matched,
                f"Response: {final_text[:500]}",
            )

        except Exception as exc:
            check(
                name,
                False,
                f"{type(exc).__name__}: {exc}",
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--live",
        action="store_true",
        help="Run real Gemini + LangGraph end-to-end scenarios",
    )

    args = parser.parse_args()

    run_deterministic_tests()

    if args.live:
        run_live_tests()

    print("\n" + "=" * 64)
    print(f"RESULT: {PASS} PASSED / {FAIL} FAILED")
    print("=" * 64)

    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
