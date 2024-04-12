import pytest

from casrl.enums.outcome import Outcome


@pytest.mark.parametrize("outcome, is_expected_terminal",[
    [Outcome.OOO, True],
    [Outcome.COL, True],
    [Outcome.NOOP, False],
])
def test_is_terminal(outcome, is_expected_terminal):
    assert is_expected_terminal == Outcome.is_terminal(outcome)
