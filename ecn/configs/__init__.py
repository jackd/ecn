import os

from absl import logging

ECN_CONFIG = os.path.realpath(os.path.dirname(__file__))

if os.environ.get("ECN_CONFIG"):
    logging.warning("ECN_CONFIG environment variable already set.")
else:
    os.environ["ECN_CONFIG"] = ECN_CONFIG
