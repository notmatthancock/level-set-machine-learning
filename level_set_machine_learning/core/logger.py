import logging
import os


DEFAULT_LOG_FILENAME = 'log.txt'


class CoreLogger(logging.Logger):
    def __init__(self, filename=None, stdout=True):
        fmt = '[%(asctime)s] %(levelname)-8s %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        # Handles when filename is None
        filename = filename or os.path.join(os.path.curdir,
                                            DEFAULT_LOG_FILENAME)

        self.file = filename
        self.stdout = stdout

        fhandler = logging.FileHandler(filename, mode='w')
        fhandler.setFormatter(formatter)

        if self.stdout:
            shandler = logging.StreamHandler()
            shandler.setFormatter(formatter)

        logging.Logger.__init__(self, 'Level set machine learning logger')
        self.setLevel(logging.DEBUG)

        self.addHandler(fhandler)

        if self.stdout:
            self.addHandler(shandler)

    def progress(self, msg, i, n):
        msg = "(%%0%dd / %d) %s" % (len(str(n)), n, msg)
        self.info(msg % i)
