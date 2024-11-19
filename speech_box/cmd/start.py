import argparse
import asyncio
import logging
from typing import Any, Dict

import yaml

from speech_box.logging import setup_logging
from speech_box.config import Config
from speech_box.server.model import ModelInstance
from speech_box.server.server import Server


logger = logging.getLogger(__name__)


class OptionalBoolAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(OptionalBoolAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs
        )
        self.default = None

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


def setup_start_cmd(subparsers: argparse._SubParsersAction):
    parser_server: argparse.ArgumentParser = subparsers.add_parser(
        "start",
        help="Run speech-box api server.",
        description="Run speech-box api server.",
    )
    group = parser_server.add_argument_group("Common settings")
    group.add_argument(
        "-d",
        "--debug",
        action=OptionalBoolAction,
        help="Enable debug mode.",
        default=False,
    )

    group = parser_server.add_argument_group("Server settings")
    group.add_argument(
        "--host",
        type=str,
        help="Host to bind the server to.",
        default="0.0.0.0",
    )
    group.add_argument(
        "--port",
        type=int,
        help="Port to bind the server to.",
        default=80,
    )
    group.add_argument(
        "--model",
        type=str,
        help="Main model path.",
        required=True,
    )
    group.add_argument(
        "--device",
        type=str,
        help="Binding device, cuda:0.",
        default="cpu",
    )
    parser_server.set_defaults(func=run)

    group = parser_server.add_argument_group("FunASR settings")
    group.add_argument(
        "--funasr-vad-model",
        type=str,
        help="Voice activity detector model path.",
        required=False,
    )
    group.add_argument(
        "--funasr-punc-model",
        type=str,
        help="Punctuation restoration model path.",
        required=False,
    )
    group.add_argument(
        "--funasr-spk-model",
        type=str,
        help="Speaker diarization model path.",
        required=False,
    )


def run(args: argparse.Namespace):
    try:
        cfg = parse_args(args)
        setup_logging(cfg.debug)
        run_model_instance(cfg)
        run_server(cfg)
    except Exception as e:
        logger.fatal(e)


def run_model_instance(cfg: Config):
    model_instance = ModelInstance(cfg)
    model_instance.run()


def run_server(cfg: Config):
    server = Server(config=cfg)

    asyncio.run(server.start())


def load_config_from_yaml(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def parse_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.debug = args.debug
    cfg.host = args.host
    cfg.port = args.port
    cfg.model = args.model
    cfg.device = args.device
    cfg.vad_model = args.funasr_vad_model
    cfg.punc_model = args.funasr_punc_model
    cfg.spk_model = args.funasr_spk_model
    return cfg


def set_config_option(args, config_data: dict, option_name: str):
    option_value = getattr(args, option_name, None)
    if option_value is not None:
        config_data[option_name] = option_value
