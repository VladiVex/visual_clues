from abc import ABC, abstractmethod
from asyncio import constants
from typing import List
from fastapi import FastAPI
from threading import Lock
import logging
import os
import sys
import urllib
import cv2
from typing import List
import shutil
# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


from common.defines import *
from common.models import ExpertParam
from common.constants import OUTPUT

from movie.movie_db import MOVIE_DB
from movie.movie_tokens import MovieTokens, TokenEntry
from movie.image_tokens import ImageTokens, ImageTokenEntry
from config.config import NEBULA_CONF


DEFAULT_FILE_PATH = "/tmp/file.mp4"
DEFAULT_FRAMES_PATH = "/tmp/movie_frames"

class BaseExpert(ABC):
    def __init__(self):
        self.db_conf = NEBULA_CONF()
        self.movie_db = MOVIE_DB()
        self.db = self.movie_db.db
        self.movie_tokens = MovieTokens(self.db)
        self.image_tokens = ImageTokens(self.db)
        self.status = ExpertStatus.STARTING
        self.tasks_lock = Lock()
        self.tasks = dict()
        self.temp_file = DEFAULT_FILE_PATH
        self.url_prefix = self.db_conf.get_webserver()

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def log_debug(self, msg):
        if self.logger:
            self.logger.debug(msg)
        else:
            print(msg)

    def log_info(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def log_error(self, msg):
        if self.logger:
            self.logger.error(msg)
        else:
            print(msg)

    def set_active(self):
        """ setting the expert status to active"""
        self.status = ExpertStatus.ACTIVE

    def add_task(self, task_id: str,  taks_params = dict()):
        with self.tasks_lock:
            self.tasks[task_id] = taks_params

    def remove_task(self, task_id: str):
        with self.tasks_lock:
            self.tasks.pop(task_id)

    @abstractmethod
    def add_expert_apis(self, app: FastAPI):
        """add expert's specific apis (REST)

        Args:
            app (FastAPI): _description_
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """return expert's name
        """
        pass

    # @abstractmethod
    def get_status(self) -> str:
        """return expert's status
        """
        return self.status.name

    def get_cfg(self) -> dict:
        """return expert's config params
        """
        return {}

    # @abstractmethod
    def get_tasks(self) -> list:
        """ return the taks currently running """
        current_tasks = list()
        with self.tasks_lock:
            for id, info in self.tasks.items():
                current_tasks.append({ 'id': id, 'info': info })
        return current_tasks

    def get_dependency(self) -> str:
        """return the expert's dependency in the pipeline:
        which pipeline step is this expert depends on
        pass

        Returns:
            str: _description_
        """
        pass

    # @abstractmethod
    def handle_msg(self, msg_params):
        """handling msg: going over the movies and calling predict on each one
        Args:
            msg_params (_type_): _description_
        """
        # output = OutputStyle.DB if not msg_params[constants.OUTPUT] else msg_params[constants.OUTPUT]
        # movies = msg_params[constants.MOVIES]
        # if isinstance(movies, list):
        #     for movie_id in movies:
        #         self.predict(movie_id, output)
        pass

    @abstractmethod
    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        pass

    @abstractmethod
    def predict_image(self, expert_params: ExpertParam):
        """ handle new movie """
        pass


    # @abstractmethod
    def handle_exit(self):
        """handle things before exit process
        """
        print(f'Exiting from: {self.get_name()}')

    def run_batch(self):
        """handle batch mode
        """
        pass

    def run_pipeline_task(self):
        """handle batch mode
        """
        pass

    # utilities
    def download_image_file(self, image_url, image_location = None, remove_prev = True):
        """ download image file to location """
        result = False
        if image_location is None:
            image_location = DEFAULT_FILE_PATH
        # remove last file
        if remove_prev and os.path.exists(image_location):
            os.remove(image_location)

        url_link = image_url
        try:
            print(url_link)
            urllib.request.urlretrieve(url_link, image_location)
            result = True
        except:
            print(f'An exception occurred while fetching {url_link}')
        return result

    def download_video_file(self, movie_id, movie_location = None, remove_prev = True):
        """download video file to location

        Args:
            movie_id (_type_): _description_
            movie_location (str): file location or default
            remove_prev: remove previous file on that location

        Returns:
            True/False
        """
        result = False
        if movie_location is None:
            movie_location = DEFAULT_FILE_PATH
        # remove last file
        if remove_prev and os.path.exists(movie_location):
            os.remove(movie_location)

        url_prefix = self.url_prefix
        url_link = ''

        movie = self.movie_db.get_movie(movie_id)
        if movie:
            try:
                url_link = url_prefix + movie['url_path']
                url_link = url_link.replace(".avi", ".mp4")
                print(url_link)
                urllib.request.urlretrieve(url_link, movie_location)
                result = True
            except:
                print(f'An exception occurred while fetching {url_link}')

        return result

    def divide_movie_into_frames(self,
        frame_list = None,
        movie_location = DEFAULT_FILE_PATH,
        movie_out_folder = DEFAULT_FRAMES_PATH,
        remove_prev = True):
        """devide move into frames

        Args:
            frame_list (_type_, optional): specific list of frames, if None then all the frames are extract. Defaults to None.
            movie_location (_type_, optional): _description_. Defaults to None.
            movie_out_folder (_type_, optional): _description_. Defaults to DEFAULT_FRAMES_PATH.

        Returns:
            _type_: list of frames saved
        """
        ret_frames = list()
        # remove last files
        if remove_prev and os.path.exists(movie_out_folder):
            shutil.rmtree(movie_out_folder, ignore_errors=True)
        os.mkdir(movie_out_folder)

        cap = cv2.VideoCapture(movie_location)

        num = 0
        while True:
            ret, frame = cap.read()
            if not cap.isOpened() or not ret:
                break
            # check for specific list to avoid saving all frames
            if frame_list and num not in frame_list:
                num = num + 1
                continue
            # save frame to file
            if frame is not None:
                frame_name = os.path.join(movie_out_folder, f'frame{num:04}.jpg')
                cv2.imwrite(frame_name, frame)
                ret_frames.append(frame_name)
            num = num + 1

        # cv2.imwrite(os.path.join(movie_out_folder, f'frame{num:04}.jpg'), frame)
        # while cap.isOpened() and ret:
        #     num = num + 1
        #     ret, frame = cap.read()
        #     if frame is not None:
        #         cv2.imwrite(os.path.join(movie_out_folder,
        #                    f'frame{num:04}.jpg'), frame)
        return ret_frames

    def save_to_db(self, movie_id, entries: List[TokenEntry]):
        error = None
        result = None
        try:
            result, error = self.movie_tokens.save_bulk_movie_tokens(movie_id, entries)
        except Exception as e:
          print(f'An exception occurred: {e}')
          error = f'execption in save_bulk_movie_tokens: {e}'
        return result, error

    def save_to_db(self, movie_id, entry: TokenEntry):
        error = None
        result = None
        try:
            result = self.movie_tokens.save_movie_token(movie_id, entry)
        except Exception as e:
          print(f'An exception occurred: {e}')
          error = f'execption in save_bulk_movie_tokens: {e}'
        return result, error

    def save_image_tokens_to_db(self, entries: List[ImageTokenEntry]):
        error = None
        result = None
        try:
            result, error = self.image_tokens.save_bulk_image_tokens(entries)
        except Exception as e:
          print(f'An exception occurred: {e}')
          error = f'execption in save_image_tokens_to_db: {e}'
        return result, error

    def save_image_token_to_db(self, entry: ImageTokenEntry):
        error = None
        result = None
        try:
            result = self.image_tokens.save_image_token(entry)
        except Exception as e:
          print(f'An exception occurred: {e}')
          error = f'execption in save_image_token_to_db: {e}'
        return result, error