import logging
import sys
import os

def config_log(name_src):
    logger = logging.getLogger(name_src)

    level = os.environ.get('LOG_LEVEL')

    logger.setLevel(logging.INFO if level is None else level)

    FORMATTER = logging.Formatter("%(asctime)s [%(name)s:%(lineno)d] %(levelname)s — %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    logger.addHandler(console_handler)

    return logger