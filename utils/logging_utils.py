import logging
import os

LOGGER_NAME = "DiffDock"
LOGLEVEL_KEY = "DIFFDOCK_LOGLEVEL"

def _get_formatter(loglevel="INFO"):
    warn_fmt = "[%(asctime)s] %(levelname)s -%(message)s"
    debug_fmt = "[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s"
    fmt = debug_fmt if loglevel.upper() in {"DEBUG", "INFO"} else warn_fmt
    return logging.Formatter(
        fmt=fmt,
        datefmt="%Y-%b-%d %H:%M:%S %Z",
    )

def remove_all_handlers(logger):
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])


def configure_logger(loglevel=None, logger_name=LOGGER_NAME, logfile=None):
    # Set as environment variable so other processes can retrieve it
    if loglevel is None:
        loglevel = os.environ.get(LOGLEVEL_KEY, "WARNING")
    else:
        os.environ[LOGLEVEL_KEY] = loglevel

    logger = logging.getLogger(logger_name)
    logger.setLevel(loglevel)

    # Ensure there is at least one handler
    if not logger.hasHandlers():
        sh = logging.StreamHandler()
        sh.setFormatter(_get_formatter(loglevel))
        logger.addHandler(sh)

    remove_all_handlers(logger)
    logger.propagate = False

    formatter = _get_formatter(loglevel)

    def _prep_handler(handler):
        handler.setLevel(loglevel)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    sh = logging.StreamHandler()
    _prep_handler(sh)

    if logfile is not None:
        fh = logging.FileHandler(logfile, mode="a")
        _prep_handler(fh)


def get_logger(base_name=LOGGER_NAME):
    """
    Return a logger.
    Use a different logger in each subprocess, though they should all have the same log level.
    """
    pid = os.getpid()
    logger_name = f"{base_name}-process-{pid}"
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        configure_logger(logger_name=logger_name)
    return logger
