import os


def get_default_config():
    """Returns the path to the defaul config file."""
    return os.path.join(os.path.dirname(__file__), 'default_config.ini')


def get_default_debug_config():
    """Return the path of the default logging config file."""
    return os.path.join(os.path.dirname(__file__), 'logging_config.ini')
