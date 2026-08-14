"""日志配置"""

import logging
import sys

from scenelog.config import LOG_LEVEL


def setup_logging():
    """配置全局日志格式和级别。"""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%H:%M:%S",
    ))

    root = logging.getLogger("scenelog")
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.addHandler(handler)

    # 抑制第三方库日志
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
