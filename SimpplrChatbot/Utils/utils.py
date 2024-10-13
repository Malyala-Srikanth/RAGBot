import logging
import sys

from API.settings import settings

LOG_LEVEL = getattr(logging, settings.LOG_LEVEL.upper())

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(levelname)s:    %(asctime)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"  # noqa: E501
)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(LOG_LEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
