"""This module is handling the logger and logging"""
# ########################
# Create the module logger
from . import (
    PerformanceWarning, search, logging,
    stdout, stderr, warnings)

'''
# Fix for the autoreload printing per reload one time the same stuff.

from scToolkit import sc_plots
from scToolkit import sc_code
from scToolkit import sc_config
from scToolkit import sc_utils
from scToolkit import sc_spatial
from scToolkit import paths
from scToolkit import sc_replacements as scr
from scToolkit import utils
from scToolkit import common_logger

scToolkit_name_to_module = {
    'sc_code': sc_code,
    "sc_utils": sc_utils,
    "utils": utils,
    "sc_config": sc_config,
    "sc_utils": sc_utils,
    "scr": scr}

for k, v in scToolkit_name_to_module.items():
    if k in globals():
        common_logger.clear_multi_loggers(v.logger)

'''


class stdout_selector(logging.Filter):
    """Basic Stderr and stdout filtering for the logging library.

    This filter allows the logging system to direct logs of specific levels
    (DEBUG, INFO, WARNING) to stdout, while other levels can be directed
    elsewhere, such as stderr.

    NOTE:
        This function is designed specifically to handle log levels
        in Python's logging library and should be integrated accordingly.

    Adopted from:
        https://stackoverflow.com/questions/16061641/python-logging-split-between-stdout-and-stderr

    Args:
        rec (logging.LogRecord): A LogRecord instance representing the log entry
            being processed.

    Returns:
        None:
            Returns True if the log level of the record is DEBUG, INFO, or
            WARNING, indicating that the record should be output to stdout.
            Returns False otherwise.
    """
    # #########################################################
    # The filter method checks the log level of the record
    def filter(
                self: logging.Filter,
                rec: logging.LogRecord
            ) -> bool:
        """
        Filters out log messages with levels DEBUG, INFO, and WARNING from stderr.

        Args:
            rec (logging.LogRecord): The log record to be checked.

        Returns:
            bool: False if the log level is DEBUG, INFO, or WARNING, True otherwise.

        TODO:
            Consider extending this function to allow customizable filtering
            based on user-defined log levels.
        """
        # ##########################################
        # Check if the log level is in the specified levels
        return rec.levelno in (logging.DEBUG, logging.INFO, logging.WARNING)


class stderr_selector(logging.Filter):
    """Basic Stderr and stdout filtering for the logging library.

    This filter is used to separate log messages based on their severity level,
    directing messages with levels DEBUG, INFO, and WARNING to stdout, and all
    others (ERROR and CRITICAL) to stderr. This is particularly useful when you
    want to keep standard output and error streams distinct.

    NOTE:
        Ensure that this filter is only applied to stderr handlers; otherwise,
        the intended separation of stdout and stderr may not work correctly.

    Adopted from:
        https://stackoverflow.com/questions/16061641/python-logging-split-between-stdout-and-stderr

    Args:
        rec (logging.LogRecord): A log record that contains all the information
            pertinent to the event being logged.

    Returns:
        None:
            Returns False if the log level is DEBUG, INFO, or WARNING, which
            prevents the log message from being directed to stderr.

    .. code-block:: python

        # To use this filter, add it to the logging configuration:
        import logging

        logger = logging.getLogger()
        stderr_handler = logging.StreamHandler(stream=sys.stderr)
        stderr_handler.addFilter(stderr_selector())
        logger.addHandler(stderr_handler)
    """
    # #########################################################
    # Function: filter
    # This function determines whether a log record should be filtered out or passed through
    # based on the log level.
    def filter(
                self: logging.Filter,
                rec: logging.LogRecord
            ) -> bool:
        """
        Filters out log messages with levels DEBUG, INFO, and WARNING from stderr.

        Args:
            rec (logging.LogRecord): The log record to be checked.

        Returns:
            bool: False if the log level is DEBUG, INFO, or WARNING, True otherwise.

        TODO:
            Consider extending this function to allow customizable filtering
            based on user-defined log levels.
        """
        # #########################################################
        # Check if the log level of the record is DEBUG, INFO, or WARNING
        # If yes, return False to filter it out from stderr.
        # ##########################################
        # This block is responsible for evaluating the log level.
        return rec.levelno not in (logging.DEBUG, logging.INFO, logging.WARNING)


def get_logger(
            name: str,
            level: str = "INFO"
        ) -> logging.Logger:
    """Create and configure a logger with the specified name and logging level.

    This function initializes a logger for the scToolkit with the given name
    and sets up the logging format to include the log level, filename, function
    name, and the log message. It configures separate handlers for standard
    output and standard error streams and applies filters accordingly. The
    function also verifies the provided logging level.

    NOTE:
        Ensure that the 'stdout_selector' and 'stderr_selector' filters are
        defined elsewhere in your codebase before using this function, as they
        are essential for filtering log messages for the respective output
        streams.

    Args:
        name (str): The name of the logger, typically representing the module or
            component.
        level (str, optional): The logging level to be set. Default is "INFO".
            Must be a valid logging level (e.g., "DEBUG", "INFO", "WARNING",
            "ERROR", "CRITICAL"). Defaults to "INFO".

    Returns:
        logging.Logger:
            The configured logger instance.

    Raises:
        ValueError: If the provided logging level is not valid.

    Tags:
        codebase, utils
    """
    # #########################################################
    # Initialize the logger with the provided name
    logger = logging.getLogger(f'scToolkit.{name}')
    # #########################################################
    # Create the formatting handler for level name, file name, and function
    # The format defines how log messages will be structured
    format = '%(levelname)s:%(filename)s:%(funcName)s - %(message)s'
    # ##########################################
    # Set up logging handlers for both stdout and stderr
    out = logging.StreamHandler(stdout)
    out.addFilter(stdout_selector())  # Applying a filter to stdout
    out.setFormatter(logging.Formatter(format))
    logger.addHandler(out)

    err = logging.StreamHandler(stderr)
    err.addFilter(stderr_selector())  # Applying a filter to stderr
    err.setFormatter(logging.Formatter(format))
    logger.addHandler(err)
    # #########################################################
    # Set the logging level based on the provided 'level' argument
    if level not in dir(logging):
        raise ValueError(f'Level {level} is not a valid logging level!')

    logger.setLevel(getattr(logging, level))
    # #########################################################
    return logger


def disable_logging(
            loggers: list[logging.Logger] | None = None,
            level: str = "CRITICAL",
            logger_name_regex: str = "scToolkit"
        ) -> None:
    """
    Disable logging for specific loggers or by library name using regular expressions.

    This function allows you to disable logging either by directly specifying
    the loggers or by using a regular expression to filter logger names. It
    provides an option to exclude specific libraries, such as "scToolkit". You
    can also specify multiple libraries to exclude by separating them with a
    pipe ("|"), like "matplotlib|scToolkit".

    NOTE:
        Useful levels are:
        "DEBUG" for the debugging logging.
        "INFO" for also informative logging.
        "CRITICAL" for ignoring most plotting.

    Args:
        loggers (list[logging.Logger] | None, optional): A list of logger
            instances to be disabled. If None, loggers are selected based on the
            ``logger_name_regex``. Defaults to None.
        level (str, optional): The logging level to set for the specified
            loggers. Useful levels are:
            "DEBUG": for debugging logging.
            "INFO": for informative logging.
            "CRITICAL": for ignoring most logging, especially during plotting.
            Default is "CRITICAL". Defaults to "CRITICAL".
        logger_name_regex (str, optional): A regular expression pattern to match
            logger names that should be disabled. Default is "scToolkit".
            Defaults to "scToolkit".

    Returns:
        None

    Raises:
        ValueError: If the specified logging level is not valid.

    Tags:
        codebase, utils
    """
    # #########################################################
    # Determine the loggers to be disabled based on input
    if loggers is None:
        # ##########################################
        # If no loggers are provided, match loggers using the regex
        if len(logger_name_regex) != 0:
            loggers = [logging.getLogger(name)
                       for name in logging.root.manager.loggerDict
                       if bool(search(logger_name_regex, name))]
        else:
            loggers = [logging.getLogger(name)
                       for name in logging.root.manager.loggerDict]
    # #########################################################
    # Validate the logging level input
    if level not in dir(logging):
        raise ValueError(f'Level {level} is not a valid logging level!')
    # #########################################################
    # Set the logging level for each logger in the list
    for inst in loggers:
        if isinstance(inst, logging.Logger):
            inst.setLevel(getattr(logging, level))


def disable_most_library_logging(message: str | None = None) -> None:
    """Set the most annoying warnings to ignore.

    This function configures the logging and warning system to ignore specific
    categories of warnings that are deemed less critical, particularly for
    collaborative work in Jupyter notebooks. The function primarily filters out
    warnings related to future deprecations and performance, which can clutter
    the output and distract from the primary analysis. However, it is advisable
    to review these warnings during development to ensure that they do not
    indicate underlying issues that need attention.

    NOTE:
        - You should not include this by default; this is for notebooks to send
          to collaborators. You should actually look at the warnings and try
          to understand their origin, to verify you are doing the correct
          thing.
        - If the ``message`` parameter is provided, it will be used to filter out
          warnings that match the given pattern. If ``message`` is not provided,
          a default set of warning messages will be filtered out.

    Args:
        message (str | None, optional): A custom warning message pattern to
            filter out. If not provided, a default Defaults to None.
        set of patterns will be used. Default is None.

    Returns:
        None:
            None

    Tags:
        utils
    """
    # #########################################################
    # Set specific warnings to be ignored
    # Ignore warnings about future deprecations
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Ignore warnings related to performance issues
    warnings.simplefilter(action='ignore', category=PerformanceWarning)
    # #########################################################
    # Filter out unwanted logger outputs related to deprecated methods and other known issues
    # Use the provided message pattern or the default one if None
    if message is None:
        # ##########################################
        # Default message pattern for filtering warnings
        message = (".*deprecated.*|.*invalid.*|.*potential.*|.*Variable names are not unique.*|"
                   ".*Geneset identifier.*|.*FutureWarning.*|.*PerformanceWarning.*")
    warnings.filterwarnings("ignore", message=message)


def clear_multi_loggers(logger: logging.Logger) -> None:
    """Clears multiple imported loggers for notebooks if %autoreload is used.

    This function is designed to reset the logging configuration in a Jupyter
    Notebook environment where the %autoreload extension is used. The autoreload
    extension can lead to the logger being imported multiple times, resulting in
    multiple handlers being added. This function checks if the provided logger
    has any existing handlers and clears them to prevent duplicate logging
    outputs.

    NOTE:
        This function is specifically useful in a Jupyter Notebook environment
        where %autoreload is used to automatically reload modules. Without
        clearing the loggers, the notebook may accumulate multiple logging
        handlers, leading to repeated log outputs.

    Args:
        logger (logging.Logger): The logger instance that needs its handlers
            cleared. This should be a logger object created using Python's
            logging module.

    Returns:
        None: This function does not return any value. It modifies the logger
            object in place.

    Tags:
        utils
    """
    # #########################################################
    # Check if the logger has any existing handlers
    if logger.hasHandlers():
        # ##########################################
        # Clear all existing handlers from the logger
        logger.handlers.clear()
