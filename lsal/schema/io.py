import abc
import logging

from lsal.utils import FilePath, file_exists, get_extension


class FileLoader(abc.ABC):
    def __init__(
            self, name: str, allowed_format: list[str], desired_outputs: list[str],
            loaded: dict = None,
    ):
        if desired_outputs is None:
            desired_outputs = []
        self.desired_outputs = desired_outputs
        if loaded is None:
            loaded = dict()
        self.loaded = loaded
        self.allowed_format = allowed_format
        self.name = name

    @abc.abstractmethod
    def load_file(self, *args, **kwargs):
        pass

    def pre_check(self, fn: FilePath):
        assert file_exists(fn), "cannot access or file does not exist: {}".format(fn)
        ext = get_extension(fn)
        assert ext in self.allowed_format, "extension not allowed: {} -- {}".format(ext, self.allowed_format)

    def post_check(self):
        assert set(self.loaded.keys()) == set(self.desired_outputs)

    def load(self, fn: FilePath, *args, **kwargs):
        logging.info("FILE LOADER: \n\t{}".format(self.__class__.__name__))
        logging.info("FILE LOADER DETAILS: \n\t{}".format(self.__class__.__doc__.strip()))
        logging.info("LOADING FILE: \n\t{}".format(fn))
        self.pre_check(fn)

        loaded = self.load_file(fn, *args, **kwargs)
        self.loaded = loaded
        logging.info("LOADED: \n\t{}".format("\n\t".join([d.__repr__() for d in self.loaded])))

        self.post_check()
        logging.info("LOADING FINISHED")
        return self.loaded
