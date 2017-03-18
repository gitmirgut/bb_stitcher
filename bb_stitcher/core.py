import os

def get_default_config():
    """
    Returns the path to the defaul config file.
    """
    return os.path.join(os.path.dirname(__file__), 'default_config.ini')


if __name__ == '__main__':
    print(get_default_config())