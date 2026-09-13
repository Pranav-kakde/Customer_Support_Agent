import time


def is_quota_error(error):
    message = str(error).lower()

    return (
        "429" in message
        or "resource_exhausted" in message
        or "quota" in message
        or "rate limit" in message
    )


def is_temporary_error(error):
    message = str(error).lower()

    temporary_keywords = [
        "503",
        "service unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection error",
        "temporarily unavailable",
    ]

    return any(keyword in message for keyword in temporary_keywords)


def get_user_friendly_error(error):
    if is_quota_error(error):
        return (
            "⚠️ The AI service has temporarily reached its usage limit. "
            "Please wait a little and try again."
        )

    if is_temporary_error(error):
        return (
            "⚠️ The AI service is temporarily unavailable. "
            "Please try again in a moment."
        )

    return (
        "⚠️ Something went wrong while processing your request. "
        "Please try again. If the problem continues, you can contact TechNova Support."
    )


def retry_with_backoff(function, max_retries=3):
    last_error = None

    for attempt in range(max_retries):
        try:
            return function()

        except Exception as error:
            last_error = error

            # Never retry quota errors
            if is_quota_error(error):
                raise error

            if not is_temporary_error(error):
                raise error

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise last_error