import pytest

from hermes_vk_community.errors import VkSecurityError
from hermes_vk_community.security import canonical_host, host_matches, validate_global_address, validate_https_url


def test_label_aware_suffix_matching() -> None:
    assert host_matches("lp.vk.com", ("vk.com",))
    assert not host_matches("evilvk.com", ("vk.com",))


@pytest.mark.parametrize(
    "url",
    [
        "http://lp.vk.com/path",
        "https://user:pass@lp.vk.com/path",
        "https://127.0.0.1/path",
        "https://vk.com.evil.test/path",
        "https://lp.vk.com.:443/path",
    ],
)
def test_long_poll_url_rejections(url: str) -> None:
    with pytest.raises(VkSecurityError):
        validate_https_url(url, suffixes=("vk.com",))


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "100.64.0.1"])
def test_non_global_addresses_are_rejected(address: str) -> None:
    with pytest.raises(VkSecurityError):
        validate_global_address(address)


def test_host_canonicalization() -> None:
    assert canonical_host("LP.VK.COM") == "lp.vk.com"
