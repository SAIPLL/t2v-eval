"""
Shared pytest fixtures for T2V logging tests.

Each test gets an isolated log directory via the ``log_dir`` fixture.
T2V logging is activated once (monkey-patches persist across tests) but the
log directory is reset to a fresh ``tmp_path`` for every test, ensuring
complete isolation between test cases.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before any plt import

import matplotlib.pyplot as plt
import pytest

from t2v_eval.logging import activate_t2v_logging


@pytest.fixture
def log_dir(tmp_path):
    """
    Activate T2V logging to an isolated temporary directory.

    Yields the log-directory path as a string.  All figures are closed after
    the test to prevent state leaking between parametrised cases.
    """
    activate_t2v_logging(str(tmp_path))
    yield str(tmp_path)
    plt.close("all")
