import logging
import os

from termcolor import colored
from tqdm import tqdm


class TqdmStream:
    @classmethod
    def write(cls, msg):
        tqdm.write(msg, end='')


class CustomFormatter(logging.Formatter):
    def __init__(self, use_colours=True):
        super().__init__()
        self.use_colours = use_colours

    def colourify(self, text, color=None, on_color=None, attrs=None):
        if not self.use_colours:
            return text
        return colored(text, color, on_color, attrs)

    def formatMessage(self, record: logging.LogRecord):
        # Create a prefix based on the log level
        if record.levelno == logging.FATAL:
            prefix = self.colourify('F>', 'red', attrs=['bold'])
        elif record.levelno == logging.CRITICAL:
            prefix = self.colourify('C>', 'red', attrs=['bold'])
        elif record.levelno == logging.ERROR:
            prefix = self.colourify('E>', 'red', attrs=['bold'])
        elif record.levelno in {logging.WARNING, logging.WARN}:
            prefix = self.colourify('W>', 'yellow', attrs=['bold'])
        elif record.levelno == logging.INFO:
            prefix = self.colourify('I>', 'blue', attrs=['bold'])
        elif record.levelno == logging.DEBUG:
            prefix = self.colourify('D>', 'green', attrs=['bold'])
        else:
            prefix = self.colourify('?>', attrs=['bold'])
        # Include logger name for child loggers only
        if record.name.find('.') >= 0:
            prefix += ' {}:'.format(record.name)
        # Add prefix to every line
        msg = '\n'.join([
            ' '.join([prefix, line])
            for line in record.getMessage().splitlines(keepends=False)
        ])
        return msg


def _setup_logger():
    """Create a logger that plays nicely with tqdm progress indicators."""
    logger = logging.getLogger('torchdata')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(TqdmStream())
    use_colours = True
    if os.environ.get('TERM', '').lower() == 'dumb' or os.environ.get('NO_COLOR', ''):
        use_colours = False
    handler.setFormatter(CustomFormatter(use_colours))
    logger.addHandler(handler)
    return logger


def create_file_handler(filename):
    handler = logging.FileHandler(filename)
    handler.setFormatter(CustomFormatter(use_colours=False))
    return handler


log = _setup_logger()
