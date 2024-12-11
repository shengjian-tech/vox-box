import logging
from vox_box.config.config import Config
import uvicorn

from vox_box.logging import setup_logging
from vox_box.server.app import app

logger = logging.getLogger(__name__)


class Server:
    def __init__(self, config: Config):
        self._config: Config = config

    @property
    def config(self):
        return self._config

    async def start(self):
        logger.info("Starting Vox Box server.")

        port = 80
        if self._config.port:
            port = self._config.port
        host = "0.0.0.0"
        if self._config.host:
            host = self._config.host

        # Start FastAPI server
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            access_log=False,
            log_level="error",
        )

        setup_logging()

        logger.info(f"Serving on {config.host}:{config.port}.")
        server = uvicorn.Server(config)
        await server.serve()
