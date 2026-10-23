"""
CLI entrypoint for NeuraCommand.
"""

import sys
from typing import Optional, Sequence

from app.commands.command_handler import CommandHandler
from app.menu.intro_menu.initial_menu import InitialMenu


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the NeuraCommand CLI.

    Args:
        argv: Optional arguments excluding the program name.

    Returns:
        Process exit code.
    """
    if argv is None:
        effective_argv = sys.argv
    else:
        effective_argv = ["neura_command", *argv]

    previous_argv = sys.argv
    sys.argv = effective_argv

    try:
        if len(sys.argv) == 1:
            InitialMenu().run()
            return 0

        handler = CommandHandler()
        handler.execute_command()
        return 0
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        sys.argv = previous_argv
