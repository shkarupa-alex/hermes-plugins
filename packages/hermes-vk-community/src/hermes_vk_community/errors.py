from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class VkApiError(Exception):
    code: int
    message: str

    def __str__(self) -> str:
        return f"VK API error {self.code}: {self.message}"


class VkSecurityError(ValueError):
    pass


class VkDeliveryUnknownError(TimeoutError):
    pass


class VkLongPollProtocolError(Exception):
    """A malformed VK Long Poll response that may recover on the next request."""


@dataclass(slots=True)
class VkHttpError(Exception):
    status: int
    operation: str

    def __str__(self) -> str:
        return f"VK {self.operation} HTTP error {self.status}"
