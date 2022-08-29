import abc
import inspect
import os

from loguru import logger
from monty.json import MSONable

from lsal.utils import FilePath, get_timestamp, createdir


class Worker(MSONable, abc.ABC):

    def __init__(self, name: str, code_dir: FilePath, work_dir: FilePath):
        self.name = name
        self.work_dir = work_dir
        self.code_dir = code_dir

    def run(self, task_names: list[str], log_file: FilePath = None):
        attrs = [getattr(self, name) for name in task_names if name in dir(self)]
        assert len(attrs) == len(task_names), f'method not found: {set(attrs) - set(task_names)}'

        createdir(self.work_dir)
        if log_file is None:
            log_file = f"{self.code_dir}/{self.name}-{get_timestamp()}.log"
        log_sink = logger.add(log_file)
        whereami = os.getcwd()
        os.chdir(self.work_dir)
        for method in attrs:
            if not inspect.ismethod(method):
                continue
            method()
        os.chdir(whereami)
        logger.remove(log_sink)
