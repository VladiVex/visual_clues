# from typing import Optional
from tkinter.messagebox import NO
from fastapi import FastAPI
# tags_metadata = [
#     {
#         "name": "status",
#         "description": "View running status of pipeline steps.",
#     },
#     {
#         "name": "set",
#         "description": "Set a configuration: <cfg_name>=<value> where cfg_name is one of the configurations \
#         run 'cfg' command to see possible configurations.",
#     },
#     {
#         "name": "cfg",
#         "description": "list all editable configurations.",
#     },
# ]

# app = FastAPI(openapi_tags=tags_metadata)

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/status")
# def get_status(q: Optional[str] = None):
#     return {"status": "running" }

# @app.post("/set}", tags=['set'] )
# def post_status(q: Optional[str] = None):
#     return {"set": 'ok', "q": q }

# @app.get("/cfg}", tags=['cfg'] )
# def get_config(q: Optional[str] = None):
#     return {"set": 'ok', "q": q }

import sys
import os
from typing import Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experts.service.base_expert import BaseExpert
from experts.app import ExpertApp
from experts.common.defines import *
from experts.common.models import ExpertParam
from experts.pipeline.api import *

# sys.path.insert(0, 'nebula3_vlm/')

class MyExpert(BaseExpert):
    def __init__(self):
        super().__init__()
        # after init all
        self.set_active()

    def get_name(self):
        return "MyExpert"

    def add_expert_apis(self, app: FastAPI):
        @app.get("/my-expert")
        def get_my_expert(q: Optional[str] = None):
            return {"expert": "my-expert" }

    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        movie = self.movie_db.get_movie(expert_params.id)
        print(f'Predicting movie: {expert_params.id}')
        return { 'result': { 'movie_id' : expert_params.id, 'info': movie , 'extra_params': expert_params.extra_params} }

    def predict_image(self, expert_params: ExpertParam):
        """ handle new image """
        result = self.download_image_file(expert_params.img_url)
        print(f'Predicting image: {expert_params.id}')
        return { 'result': result, 'image_id' : expert_params.id, 'url': expert_params.img_url , 'extra_params': expert_params.extra_params}

    def run_batch(self):
        """handle batch mode
        """
        print(f'running batch from: {self.get_name()}')

    def run_pipeline_task(self):
        print(f'running pipeline task from: {self.get_name()}')
        pipeline = PipelineApi(None)
        pipeline_doc = pipeline.get_first_pipeline_for_task('videoprocessing')
        print(pipeline_doc)
        # now update a d remove videoprocessing from the pipeline doc
        if pipeline_doc:
            pipeline.update_pipeline(pipeline_id=pipeline_doc['_key'],
                                    movies_status=None,
                                    task_status={'videoprocessing': STATUS_ACTIVE },
                                    task_inputs={})

my_expert = MyExpert()
expert_app = ExpertApp(expert=my_expert)
app = expert_app.get_app()
expert_app.run()
