"""Configuration display command."""

from __future__ import annotations

import click

from guidellm.settings import print_config

__all__ = ["config"]


@click.command(
    short_help="Show configuration settings.",
    help="Display environment variables for configuring GuideLLM behavior.",
)
def config():
    print_config()
