import time
from logging import Logger


class timer:
    """
    util function used to log the time taken by a part of program
    """

    def __init__(self, logger: Logger, message: str):
        """
        init the timer

        Parameters
        ----------
        logger: Logger
            logger to write the logs
        message: str
            message to log, like start xxx
        """
        self.message = message
        self.logger = logger
        self.start = 0
        self.duration = 0
        self.sub_timers = []

    def __enter__(self):
        """
        context enter to start write this
        """
        self.start = time.time()
        self.logger.info("Starting %s" % self.message)
        return self

    def __exit__(self, context, value, traceback):
        """
        context exit will write this
        """
        self.duration = time.time() - self.start
        self.logger.info(f"Finished {self.message}, that took {self.duration:.3f}")
