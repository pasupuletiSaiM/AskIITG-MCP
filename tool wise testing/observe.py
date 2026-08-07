import functools
import time

import logfire

from config import Config

# Configure Logfire-------------------------------------------
logfire.configure(
    service_name=Config.LOGFIRE_SERVICE,
    send_to_logfire="if-token-present",
    inspect_arguments=False,
    console=logfire.ConsoleOptions(min_log_level="info"),
)



def observe(tool_name: str):
    """
    Decorator to trace every MCP tool call.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            start = time.perf_counter()

            # Get query if present
            query = kwargs.get("query")
            if query is None and args:
                query = args[0]

            query = query if isinstance(query, str) else ""

            with logfire.span(
                "tool {tool}",
                tool=tool_name,
                query=query,
            ) as span:

                try:
                    result = func(*args, **kwargs)

                except Exception as e:

                    span.set_attribute("success", False)

                    span.set_attribute("error", str(e))

                    logfire.error(
                        "Tool {tool} failed",
                        tool=tool_name,
                    )

                    raise

                latency = round(
                    (time.perf_counter() - start) * 1000,
                    2,
                )

                span.set_attribute("success", True)

                span.set_attribute("latency_ms", latency)

                logfire.info(
                    "Tool {tool} completed in {latency} ms",
                    tool=tool_name,
                    latency=latency,
                )

                return result

        return wrapper

    return decorator