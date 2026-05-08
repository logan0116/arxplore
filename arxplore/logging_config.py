"""
logging_config.py — 日志配置
"""
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """配置全局日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> logging.Logger:
    """获取模块 logger"""
    return logging.getLogger(name)
