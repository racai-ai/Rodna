import sys

_printed_errors = set()

def print_error(message: str, calling_fn: str) -> None:
    """Print a given message only once."""

    if message not in _printed_errors:
        print("{0}: {1}".format(calling_fn, message), file=sys.stderr, flush=True)
        _printed_errors.add(message)
    # end if
