import logging
import sys


def pytest_sessionstart(session):
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
