"""
author:Sanidhya Mangal, Daniel Shu, Rishav Sen, Jatin Kodali
"""

import logging
import logging.handlers
import sys

consoleLoggingLevel = logging.DEBUG

# logDir = os.path.join(os.curdir, 'log')
loggerName = 'gans'

# Make sure the logging directory exist; create it otherwise.
# if not os.path.exists(logDir):
#     os.makedirs(logDir)

logger = logging.getLogger(loggerName)
logger.setLevel(logging.DEBUG)

# Create the logging file and allow the file to rollover at a predetermined size.
# When the size is about to exceed 1 MB, the file is closed and a new file is
# silently opened for new output.
# logFilePath = logDir + '/' + loggerName + '.log'
# fh = logging.handlers.RotatingFileHandler(logFilePath,
#                                           maxBytes=1024000,
#                                           backupCount=5)
ch = logging.StreamHandler(sys.stdout)

# # Set file logging level
# fh.setLevel(fileLoggingLevel)

# Set console logging level
ch.setLevel(consoleLoggingLevel)

formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)

# logger.addHandler(fh)
logger.addHandler(ch)

logger.debug('Started logging')
