import pytest
from pydantic import ValidationError

from hermes_vk_community.models import LongPollLease, LongPollResponse


@pytest.mark.parametrize("value", [81, "81"])
def test_long_poll_lease_normalizes_numeric_cursor(value: int | str) -> None:
    lease = LongPollLease.model_validate({"key": "key", "server": "https://lp.vk.com", "ts": value})
    assert lease.ts == "81"


@pytest.mark.parametrize("value", [81, "81"])
def test_long_poll_response_normalizes_numeric_cursor(value: int | str) -> None:
    response = LongPollResponse.model_validate({"ts": value, "updates": []})
    assert response.ts == "81"


def test_long_poll_cursor_rejects_boolean() -> None:
    with pytest.raises(ValidationError):
        LongPollResponse.model_validate({"ts": True, "updates": []})
