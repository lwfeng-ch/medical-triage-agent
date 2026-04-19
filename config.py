# config.py
"""
统一配置文件 - 兼容层

此文件保持向后兼容，所有配置已迁移至 utils/config/ 目录。

新代码建议使用：
    from utils.config import Config, LLMConfig, VectorStoreConfig

旧代码可继续使用：
    from utils.config import Config
    # 或
    from config import Config
"""

from utils.config import (
    Config,
    LLMConfig,
    VectorStoreConfig,
    MiddlewareConfig,
    ServiceConfig,
    LoggingConfig,
)

__all__ = [
    "Config",
    "LLMConfig",
    "VectorStoreConfig",
    "MiddlewareConfig",
    "ServiceConfig",
    "LoggingConfig",
]
