from typing import Optional


class Config:
    """A class used to define speech-box configuration.

    Attributes:
        debug: Enable debug mode.
        host: Host to bind the server to.
        port: Port to bind the server to.
        model: Model path.
    """

    # Common options
    debug: bool = False

    # Server options
    host: Optional[str] = "0.0.0.0"
    port: Optional[int] = None

    # Model options
    model: Optional[str] = None
    device: Optional[str] = "cpu"

    # FunASR options
    vad_model: Optional[str] = None
    punc_model: Optional[str] = None
    spk_model: Optional[str] = None
