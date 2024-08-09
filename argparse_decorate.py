"""
A simple general argparse decorator.
"""
import argparse


class Arg:
    def __init__(self, function, parser):
        """
        Use parser for certain function.
        """
        self.function = function
        self.parser = parser

    def __call__(self):
        args = self.parser.parse_args()
        return self.function(**vars(args))


def add_arg(*args, **kwargs):
    """
    Add arguments for certain function.
    """
    def decorator(function):
        if not hasattr(function, 'arg'):
            function.arg = []
        function.arg.append((args, kwargs))
        return function
    return decorator


def init_parser(*args, **kwargs):
    """
    Initialize a parser and add all arguments in it.
    """
    def decorator(function):
        parser = argparse.ArgumentParser(*args, **kwargs)
        params = function.arg
        for param in params:
            parser.add_argument(*param[0], **param[1])

        return Arg(function, parser)
    return decorator
