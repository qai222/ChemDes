import abc
import inspect
import os

from loguru import logger
from monty.json import MSONable

from lsal.utils import FilePath, get_timestamp, createdir, copyfile, get_file_size, json_dump


class Worker(MSONable, abc.ABC):

    def __init__(self, name: str, code_dir: FilePath, work_dir: FilePath, collect_files: list[FilePath] = None):
        self.name = name
        self.work_dir = work_dir
        self.code_dir = code_dir
        if collect_files is None:
            collect_files = []
        self.collect_files = collect_files

    @property
    def worker_json(self):
        return os.path.join(self.code_dir, f"{self.name}.json")

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
        json_dump(self, self.worker_json)
        os.chdir(whereami)
        logger.remove(log_sink)

    def final_collect(self):
        warning_file_size = 45.0
        warning_file_size_unit = "m"
        if len(self.collect_files) > 0:
            logger.info(f"Finally, copy the following files to: {self.code_dir}")
        for fn in self.collect_files:
            logger.info(f"copying: {fn}")
            if get_file_size(fn, warning_file_size_unit) > warning_file_size:
                logger.warning(
                    f"files size larger than {warning_file_size} {warning_file_size_unit}, please double check: {fn}")
            copyfile(fn, self.code_dir)
            logger.info(f"file copied: {fn}")
