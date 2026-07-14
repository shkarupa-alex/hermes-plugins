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
