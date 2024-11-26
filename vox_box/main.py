import argparse
import signal
import sys

from vox_box.cmd import setup_start_cmd
from vox_box.cmd.version import setup_version_cmd


def handle_signal(sig, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def main():
    parser = argparse.ArgumentParser(
        description="VoxBox",
        conflict_handler="resolve",
        add_help=True,
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=55, indent_increment=2, width=200
        ),
    )
    subparsers = parser.add_subparsers(help="sub-command help")

    setup_start_cmd(subparsers)
    setup_version_cmd(subparsers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
