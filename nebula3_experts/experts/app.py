from cmath import exp
from functools import cache
import os
import sys
from queue import Queue
import logging
import json
import uuid
from threading import Thread, Lock
from typing import List
from fastapi import FastAPI
from numpy import array
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from cachetools import TTLCache

# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experts.service.base_expert import BaseExpert
from experts.pipeline.api import PipelineApi
import experts.common.constants as constants
from experts.common.defines import ExpertCommands, OutputStyle, RunMode
from experts.common.models import ExpertParam
from experts.common.config import ExpertConf


tags_metadata = [
    {
        "name": "status",
        "description": "View running status of pipeline steps.",
    },
    {
        "name": "set",
        "description": "Set a configuration: <cfg_name>=<value> where cfg_name is one of the configurations \
        run 'cfg' command to see possible configurations.",
    },
    {
        "name": "cfg",
        "description": "list all editable configurations.",
    },
    {
        "name": "jobs",
        "description": "list all jobs (running/finish - cached) and their status.",
    },
    {
        "name": "predict",
        "description": "run the expert on the specific movie and location. with the given output",
    },
]

class PredictParam(BaseModel):
    movie_id: str
    scene_element: int = None
    local: bool
    extra_params: dict = None
    output: str = constants.OUTPUT_JSON

    class Config:
        schema_extra = {
            "example": {
                "movie_id": "the movie id in db",
                "scene_element": "movie's scene element",
                "local": "movie location: local (true) /remote (false)",
                "extra_params": "the expert's specific params in json object",
                "output": "where to output: json (return json in response)/db, default- db"
            }
        }

class PredictImageParam(BaseModel):
    image_id: str
    url: str
    extra_params: dict = None
    output: str = constants.OUTPUT_JSON
    output_file: str = None
    is_async: bool = False

    class Config:
        schema_extra = {
            "example": {
                "image_id": "The image id",
                "url": "url to the image (remote: 'http://..' , local: 'file://...')",
                "extra_params": "the expert's specific params in json object",
                "output": "where to output: json (return json in response)/db/file, default- json",
                "output_file": "when output is 'file' => the path of the output file e.g: /mnt/images/image1.json",
                "is_async": "will this api runs async, default: False"
            }
        }

class ImageParam(BaseModel):
    image_id: str
    url: str

class PredictImagesParam(BaseModel):
    images: List[ImageParam]
    extra_params: dict = None
    output: str = constants.OUTPUT_JSON
    output_file: str = None
    is_async: bool = False

    class Config:
        schema_extra = {
            "example": {
                "image_id": "The image id",
                "url": "url to the image (remote: 'http://..' , local: 'file://...')",
                "extra_params": "the expert's specific params in json object",
                "output": "where to output: json (return json in response)/db/file, default- json",
                "output_file": "when output is 'file' => the path of the output file e.g: /mnt/images/image1.json",
                "is_async": "will this api runs async, default: False"
            }
        }


class ExpertApp:
    def __init__(self, expert: BaseExpert, params = None):
        self.params = params
        self.expert = expert
        self.running = True
        self.logger = self.init_logger()
        self.msgq = Queue()
        self.jobs_lock = Lock()
        self.config = ExpertConf()
        # check env for disable pipeline
        if self.config.get_run_pipeline():
            self.pipeline =  PipelineApi(self.logger) #PIPELINE_API(self.logger)
            self.init_pipeline()
        else:
            self.pipeline = None
        self.run_mode = RunMode.resolve_run_mode(self.config.get_run_mode())
        self.app = FastAPI(openapi_tags=tags_metadata)
        self.add_base_apis()
        self.expert.set_logger(self.logger)
        # Add expert specific apis
        self.expert.add_expert_apis(self.app)
        # init async
        num_of_executors = self.config.get_thread_pool_size()
        self.pool = ThreadPoolExecutor(num_of_executors)
        self.job_cache = TTLCache(maxsize=10, ttl=self.config.get_jobs_expiration()) # 10 minute cache
        # check for batch mode
        if self.run_mode == RunMode.BATCH:
            self.start_batch_mode()
        elif self.run_mode == RunMode.TASK:
            self.start_task_mode()


    def __del__(self):
        if not self.running:
            self.running = False
            self.msgq.put_nowait(constants.STOP_MSG)
            if (self.msg_thread and self.msg_thread.is_alive()):
                self.msg_thread.join()

    def init_logger(self):
        """
        configure the global logger:
        - write DEBUG+ level to stdout
        - write ERROR+ level to stderr
        - format: [time][thread name][log level]: message
        @param log_file: the file to which we wish to write. if an existing dir is given, log to a file
                        labeled with the curent date and time. if None, use the current working directory.
        """
        logger_name = f'{self.expert.get_name()}-{os.getpid()}'
        # create a logger for this instance
        logger = logging.getLogger(logger_name)

        # set general logging level to debug
        logger.setLevel(logging.DEBUG)

        # choose logging format
        formatter = logging.Formatter('[%(asctime)s][%(threadName)s][%(levelname)s]: %(message)s')

        # create stdout stream handler
        shdebug = logging.StreamHandler(sys.stdout)
        shdebug.setLevel(logging.DEBUG)
        shdebug.setFormatter(formatter)
        logger.addHandler(shdebug)

        # create stderr stream handler
        sherr = logging.StreamHandler(sys.stderr)
        sherr.setLevel(logging.ERROR)
        sherr.setFormatter(formatter)
        logger.addHandler(sherr)

        title = f'===== {logger_name} ====='
        logger.info('=' * len(title))
        logger.info(title)
        logger.info('=' * len(title))

        return logger

    def init_pipeline(self):
        """init pipeline params
        and add pipeline event handlers/callbacks
        """
        self.pipeline.subscribe(self.expert.get_name(),
                                self.expert.get_dependency(),
                                self.on_pipeline_msg)
        self.msg_thread = Thread(target=self.msg_handle)
        # TODO enable this for pipeline
        # self.msg_thread.start()

    def on_pipeline_msg(self, msg):
        """handle msg received from pipeline subscription
        Args:
            msg (_type_): msg received
        """

        # log msg put to queue
        self.msgq.put_nowait({ constants.COMMAND: ExpertCommands.PIPELINE_NOTIFY, constants.PARAMS: msg })

    def msg_handle(self):
        """handle incoming msg
        """
        msg_iterator = iter(self.msgq, constants.STOP_MSG)
        while self.running:
            try:
                msg = next(msg_iterator)
            except StopIteration:
                # log exit
                if self.expert.handle_exit:
                    self.expert.handle_exit()
                self.running = True
            else:
                # log msg received
                cmd = msg[constants.COMMAND]
                params = msg[constants.PARAMS]
                if ExpertCommands.PIPELINE_NOTIFY == cmd:
                    self.logger.debug(f'Recieved PIPELINE_NOTIFY: {params[constants.ID]}')
                    self.expert.handle_msg(params)
                elif ExpertCommands.CLI_PREDICT == cmd:
                    self.logger.debug(f'Recieved CLI_PREDICT: {params[constants.ID]}')
                    self.expert.handle_msg(params)
                else:
                    msg_json = json.dumps(msg)
                    self.logger.error(f'Received unknown: {msg_json}')

    def predict_movie(self, params: PredictParam):
        """ predict - this is an async command so putting to queue """
        # parsing into ExpertParams
        expert_params = self.parse_params(params)
        # if expert_params.output == constants.OUTPUT_DB:
        #     self.msgq.put_nowait({ constants.COMMAND: ExpertCommands.CLI_PREDICT, constants.PARAMS: expert_params })
        # no output style is like json -> predict and return result
        return self.expert.predict(expert_params)

    def predict_image(self, params: PredictImageParam):
        """ predicting image """
        # parsing into ExpertParams
        expert_params = self.parse_image_params(params)
        # if expert_params.output == constants.OUTPUT_DB: currently not queuing
        #     self.msgq.put_nowait({ constants.COMMAND: ExpertCommands.CLI_PREDICT, constants.PARAMS: expert_params })
        # no output style is like json -> predict and return result
        # getting the image
        job_id = str(uuid.uuid4())
        with self.jobs_lock:
            self.job_cache[job_id] = { "state": "running", "params" :expert_params }
        future = self.pool.submit(self.handle_predict_image, expert_params, job_id)
        if params.is_async:
            return { 'jobId': job_id}
        else:
            response = future.result(self.config.get_max_wait_predict_time()) # self.expert.predict_image(expert_params)
            return response

    def predict_images(self, params: PredictImagesParam):
        """ predicting image """
        # parsing into ExpertParams
        expert_params_list = self.parse_images_params(params)
        # if expert_params.output == constants.OUTPUT_DB: currently not queuing
        #     self.msgq.put_nowait({ constants.COMMAND: ExpertCommands.CLI_PREDICT, constants.PARAMS: expert_params })
        # no output style is like json -> predict and return result
        # getting the image
        job_id = str(uuid.uuid4())
        with self.jobs_lock:
            self.job_cache[job_id] = { "state": "running", "params": expert_params_list }
        future = self.pool.submit(self.handle_predict_image_list, expert_params_list, job_id)
        if params.is_async:
            return { 'jobId': job_id}
        else:
            response = future.result(self.config.get_max_wait_predict_time()) # self.handle_predict_image_list(expert_params_list, job_id)
            return response

    def add_base_apis(self):
        """add base apis
        Returns:
            _type_: _description_
        """
        @self.app.get("/")
        def read_root():
            return {"Expert": self.expert.get_name()}


        @self.app.get("/status")
        def get_status():
            return {"status": self.expert.get_status()}

        @self.app.post("/set", tags=['set'] )
        def post_config():
            return {"set": 'not supported' }

        @self.app.get("/cfg", tags=['cfg'] )
        def get_config():
            return {"cfg": self.expert.get_cfg()}

        @self.app.get("/tasks", tags=['tasks'] )
        def get_tasks():
            return {"tasks": self.expert.get_tasks() }

        @self.app.get("/jobs", tags=['tasks'] )
        def get_tasks():
            return {"jobs": self.get_all_jobs() }

        @self.app.get("/jobs/{job_id}", tags=['tasks'] )
        def get_tasks(job_id: str):
            return self.get_job(job_id)

        @self.app.post("/predict", tags=['run'] )
        async def predict(params: PredictParam):
            return self.predict_movie(params)

        @self.app.post("/predict/movie", tags=['run'] )
        async def predict(params: PredictParam):
            return self.predict_movie(params)

        @self.app.post("/predict/image", tags=['run'] )
        async def predict_image(params: PredictImageParam):
            return self.predict_image(params)

        @self.app.post("/predict/images", tags=['run'] )
        async def predict_images(params: PredictImagesParam):
            return self.predict_images(params)

    def parse_params(self, params):
        expert_params = ExpertParam(params.movie_id,
                                    params.scene_element,
                                    params.local,
                                    params.extra_params,
                                    params.output)
        return expert_params
    def parse_image_params(self, params):
        expert_params = ExpertParam(params.image_id,
                                    None,
                                    False,
                                    params.extra_params,
                                    params.output,
                                    constants.TYPE_IMAGE,
                                    params.url,
                                    params.output_file)
        return expert_params
    def parse_images_params(self, params: PredictImagesParam):
        expert_params_list = list()
        for img in params.images:
            expert_params = ExpertParam(img.image_id,
                                        None,
                                        False,
                                        params.extra_params,
                                        params.output,
                                        constants.TYPE_IMAGE,
                                        img.url,
                                        params.output_file)
            expert_params_list.append(expert_params)
        return expert_params_list

    def handle_predict_image(self, expert_params: PredictImagesParam, job_id: str):
        response = self.expert.predict_image(expert_params)
        with self.jobs_lock:
            self.job_cache[job_id] = { "state": "finish", **response }
        return response

    def handle_predict_image_list(self, expert_params_list: List[PredictImagesParam], job_id: str):
        results = list()
        for expert_param in expert_params_list:
            results.append(self.expert.predict_image(expert_param))
        with self.jobs_lock:
            self.job_cache[job_id] = { "state": "finish", "results": results }
        return { 'results' : results,  'jobId': job_id }

    def get_all_jobs(self):
        jobs = list()
        with self.jobs_lock:
            for cached_item in self.job_cache.items():
                jobs.append({ "jobId": cached_item[0], **cached_item[1]})
        return jobs

    def get_job(self, job_id: str):
        job_info = None
        with self.jobs_lock:
            if self.job_cache.__contains__(job_id):
                job_info = self.job_cache.__getitem__(job_id)
                return { job_id: { **job_info }}
        return { }

    def run(self):
        print("Running...")
        # self.expert.run()

    def get_app(self):
        return self.app

    def start_batch_mode(self):
        self.logger.info("Start batch mode")
        try:
            self.expert.run_batch()
        except:
            self.logger.error("error in batch run. exiting")
            sys.exit(os.EX_SOFTWARE)
        self.logger.info("finish batch mode. exiting")
        sys.exit(os.EX_OK) # code 0, all ok

    def start_task_mode(self):
        self.logger.info("Start pipeline-task mode")
        try:
            self.expert.run_pipeline_task()
        except Exception as e:
            self.logger.error(f"error: {e} in task run. exiting")
            sys.exit(os.EX_SOFTWARE)
        self.logger.info("finish task mode. exiting")
        sys.exit(os.EX_OK) # code 0, all ok
