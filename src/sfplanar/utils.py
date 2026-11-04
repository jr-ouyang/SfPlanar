import numpy as np
from dataclasses import is_dataclass
from typing import Any

from datetime import timedelta


class ReadableStrMixin:
    def __str__(self) -> str:
        return self._str_recursively(self)

    @staticmethod
    def _str_recursively(obj: Any, indent: int = 0) -> str:
        lines = []
        prefix = " " * indent

        # obtain key–value pairs depending on object type
        if is_dataclass(obj):
            items = vars(obj).items()
        elif isinstance(obj, dict):
            items = obj.items()
        # (included for completeness, not expected in our usage)
        else:
            # other types -> return its string with indentation
            return f"{prefix}{obj}"

        # format each key–value
        #   dataclass/dict -> recursive block
        #   other types    -> inline with formatted values
        for key, value in items:
            if is_dataclass(value) or isinstance(value, dict):
                # append the key separately
                lines.append(f"{prefix}{key}:")
                # call `_str_recursively` with the value and the increased indentation
                lines.append(ReadableStrMixin._str_recursively(value, indent + 4))
            else:
                # format the leaf real numbers in a recursive manner
                fomatted_value = ReadableStrMixin._format_leaf_Real_recursively(value)
                lines.append(f"{prefix}{key}: {fomatted_value}")

        return "\n".join(lines)

    @staticmethod
    def _format_leaf_Real_recursively(obj: Any, num_sig_digits=7) -> Any:
        # local alias
        _format_leaf_Real_recursively = ReadableStrMixin._format_leaf_Real_recursively

        # float -> format with the given num_sig_digits
        if isinstance(obj, (float, np.floating)):
            return f"{obj:,.{num_sig_digits}g}"
        # integer -> format with Thousands Separators
        elif isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
            return f"{obj:,d}"
        # tuple -> format the leaf real number recursively
        elif isinstance(obj, tuple):
            items = (_format_leaf_Real_recursively(item, num_sig_digits) for item in obj)
            inner = ", ".join(items)
            return "(" + inner + ")"
        # list -> format the leaf real number recursively
        elif isinstance(obj, list):
            items = (_format_leaf_Real_recursively(item, num_sig_digits) for item in obj)
            inner = ", ".join(items)
            return "[" + inner + "]"
        # dict -> format the leaf real number recursively
        # (included for completeness, not expected in our usage)
        elif isinstance(obj, dict):
            items = (f"{k}: {_format_leaf_Real_recursively(v, num_sig_digits)}" for k, v in obj.items())
            inner = ", ".join(items)
            return "{" + inner + "}"
        # np.ndarray -> return its type and shape
        elif isinstance(obj, np.ndarray):
            return f"{type(obj).__name__} of shape {obj.shape}"
        # other types -> return as is
        else:
            return obj


def format_runtime(runtime, digits=1):
    # Get the integer part and the fractional part
    runtime_int = int(runtime)
    runtime_frac = runtime - runtime_int

    # Scale fractional part and round to desired number of digits
    frac = round(runtime_frac * 10**digits)

    # Handle overflow
    if frac == 10**digits:
        runtime_int += 1
        frac = 0

    formatted_runtime = str(timedelta(seconds=runtime_int))
    if frac:
        formatted_runtime += f".{frac:0{digits}d}"
    return formatted_runtime
