"""infra 统一日志封装。

基于标准库 logging（零额外依赖），把 infra 内的 print 收敛为 logger 调用，
输出到文件（可选同时到控制台）。

用法：
    from src.infra.log import get_logger

    logger = get_logger(__name__)
    logger.info("...")
    logger.warning("...")

入口处调用一次 setup_logging 指定日志文件：
    from src.infra.log import setup_logging
    setup_logging("outputs/infra.log")
"""

import logging
import os
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    log_file: Optional[str] = "outputs/infra.log",
    level: int = logging.INFO,
    console: bool = True,
) -> None:
    """初始化根 logger：写文件（可选同时到控制台）。幂等，重复调用不会叠加 handler。"""
    root = logging.getLogger()
    if root.handlers:
        return  # 已配置过

    root.setLevel(level)
    formatter = logging.Formatter(_DEFAULT_FORMAT)

    if log_file is not None:
        parent = os.path.dirname(log_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """获取以模块名为名的 logger。"""
    return logging.getLogger(name)
