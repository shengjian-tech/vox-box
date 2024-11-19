import argparse
import json
import logging
import os
import sys

from speech_box.cmd.start import OptionalBoolAction, run_model_instance
from speech_box.config.config import Config
from speech_box.logging import setup_logging
from speech_box.server.model import get_model_instance
from speech_box.utils.model import create_model_dict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def setup_estimate_cmd(subparsers: argparse._SubParsersAction):
    parser_server: argparse.ArgumentParser = subparsers.add_parser(
        "estimate",
        help="Run the speech-box for model estimate.",
        description="Run the speech-box for model estimate.",
    )
    parser_server.add_argument(
        "-d",
        "--debug",
        action=OptionalBoolAction,
        help="Enable debug mode.",
        default=False,
    )

    parser_server.add_argument(
        "--model",
        type=str,
        help="Main model path.",
        required=True,
    )

    parser_server.set_defaults(func=run)


def run(args: argparse.Namespace):
    result = {}
    with suppress_output():
        try:
            cfg = parse_args(args)
            setup_logging(cfg.debug)
            run_model_instance(cfg)

            instance = get_model_instance()
            model_info = create_model_dict(cfg.model, supported=False)
            if instance is not None:
                model_info = instance.model_info()

            result = model_info
        except Exception as e:
            result = {
                "error": str(e),
            }
    print(json.dumps(result, indent=4, ensure_ascii=False))


def parse_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.debug = args.debug
    cfg.model = args.model
    return cfg


@contextmanager
def suppress_output():
    null_writer = open(os.devnull, "w")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = null_writer
    sys.stderr = null_writer
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        null_writer.close()
