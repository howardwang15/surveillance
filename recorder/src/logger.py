import logging
import sys
import os
from logging.handlers import RotatingFileHandler

class Logger:
    def __init__(self, name, log_path, default_level, max_size, num_files):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(default_level)
        formatter = logging.Formatter('%(asctime)s %(message)s')

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = RotatingFileHandler(log_path, maxBytes=max_size, backupCount=num_files)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)

    def write(self, level, message):
        self.logger.log(level, message)
