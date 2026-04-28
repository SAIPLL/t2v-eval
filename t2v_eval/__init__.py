from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("t2v_eval")
except PackageNotFoundError:
    # Package not installed (e.g. running directly from source)
    __version__ = "unknown"
