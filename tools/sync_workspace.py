"""Install the locked workspace and a frozen, editable Hermes source checkout."""

from __future__ import annotations
import argparse
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / ".hermes-agent"
SPARSE_DIRECTORIES = (
    "hermes_platform",
    "agent",
    "tools",
    "hermes_cli",
    "gateway",
    "pm",
    "tui_gateway",
    "cron",
    "acp_adapter",
    "plugins",
    "providers",
    "locales",
    "skills",
    "optional-skills",
    "optional-mcps",
)


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)  # noqa: S603 - fixed executables and repository-owned arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", help="Install one plugin instead of the whole workspace")
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    source = project["tool"]["uv"]["sources"]["hermes-agent"]
    # Hermes intentionally refuses ordinary wheel builds: its runtime needs the
    # source layout. Keep the Git source in uv.lock for resolution/audit, but
    # install the same immutable revision through its supported editable path.
    if not (HOST / ".git").is_dir():
        run("git", "init", str(HOST))
        run("git", "-C", str(HOST), "remote", "add", "origin", source["git"])
    run("git", "-C", str(HOST), "sparse-checkout", "set", *SPARSE_DIRECTORIES)
    current = subprocess.run(  # noqa: S603 - read-only Git probe of the owned checkout
        [shutil.which("git") or "git", "-C", str(HOST), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if current.stdout.strip() != source["rev"]:
        run("git", "-C", str(HOST), "fetch", "--depth", "1", "--filter=blob:none", "origin", source["rev"])
        run("git", "-C", str(HOST), "checkout", "--detach", source["rev"])
    common = ("--locked", "--no-install-package", "hermes-agent")
    if args.package:
        # --package selects the member's groups, so explicitly install the
        # root test tools first, then retain them while adding that member.
        run("uv", "sync", "--only-dev", *common)
        run("uv", "sync", "--package", args.package, "--inexact", *common)
    else:
        run("uv", "sync", "--all-packages", *common)
    run("uv", "pip", "install", "--no-deps", "--editable", str(HOST))


if __name__ == "__main__":
    main()
