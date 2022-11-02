import os
import sys
import time
import threading
import logging
from typing import Tuple
from abc import ABC, abstractmethod
# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
# from movie.movie_db import MOVIE_DB
# from config.config import  NEBULA_CONF

from nebula3_database.movie_db import MOVIE_DB
from nebula3_database.config import NEBULA_CONF

COLLECTION_NAME = 'pipelines'
PIPELINE_MOVIES = 'movies'
PIPELINE_TASKS = 'tasks'
PIPELINE_INPUTS = 'inputs'
TASK_STATUS = 'status'
STATUS_ACTIVE = 'active'
STATUS_SUCCESS = 'success'
STATUS_FAIL = 'fail'


class PipelineTask(ABC):
    @abstractmethod
    def process_movie(self, movie_id: str) -> Tuple[bool, str]:
        """process movie and return True/False and error str if False
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class PipelineApi:
    def __init__(self, logger: logging.Logger):
        self.config = NEBULA_CONF()
        self.movie_db = MOVIE_DB()
        self.db = self.movie_db.db
        self.running = True
        self.logger = logger
        self.subscriptions = list()
        # self.database = config.get_database_name()
        # self.dbconn = DatabaseConnector()
        # self.db = self.dbconn.connect_db(self.database)
        # self.movie_db = MOVIE_DB(self.db)

    # def __del__(self):
    #     self.running = False
    #     for sub_thread in self.subscriptions:
    #         if (sub_thread.is_alive()):
    #             sub_thread.join()

    # def subscription_loop(self, entity_name: str, entity_dependency: str, msg_cb):
    #     while self.running:
    #         # Signaling your code, that we have newly uploaded movie, frames are stored in S3.
    #         # Returns movie_id in form: Movie/<xxxxx>
    #         movies = self.wait_for_change(entity_name, entity_dependency)
    #         for movie in movies:
    #             msg_cb(movie)
    #         self.update_expert_status(entity_name) #Update scheduler, set it to done status

    # def subscribe(self, entity_name: str, entity_dependency: str, msg_cb):
    #     """subscribe for msgs

    #     Args:
    #         entity_name (str): the pipeline entity name
    #         entity_dependency (str): the pipeline entity dependency
    #         msg_cb (_type_): _description_
    #     """
    #     sub_thread = threading.Thread(target=self.event_loop,
    #                                          args=[entity_name, entity_dependency, msg_cb])
    #     sub_thread.start()
    #     self.subscriptions.append(sub_thread)

    # def get_new_movies(self):
    #     return self.movie_db.get_new_movies()

    # def get_all_movies(self):
    #     return self.movie_db.get_all_movies()

    # def get_versions(self):
    #     versions = []
    #     query = 'FOR doc IN changes RETURN doc'
    #     cursor = self.db.aql.execute(query)
    #     for data in cursor:
    #         #print(data)
    #         versions.append(data)
    #     return(versions)

    # def get_expert_status(self, expert, depends):
    #     versions = self.get_versions()
    #     for version in versions:
    #         if version[depends] > version[expert]:
    #             #print(version)
    #             return True
    #         else:
    #             return False

    # def wait_for_change(self, expert, depends):
    #     while True:
    #         if self.get_expert_status(expert, depends):
    #             movies = self.get_new_movies()
    #             #print("New movies: ", movies)
    #             return(movies)
    #         time.sleep(3)

    # def wait_for_finish(self, experts):
    #     while True:
    #         versions = self.get_versions()
    #         count = len(experts)
    #         for version in versions:
    #             global_version = version['movies']
    #             print(version)
    #             for expert  in experts:
    #                 if global_version != version[expert]:
    #                     break
    #                 else:
    #                     count = count - 1
    #         if count <= 0:
    #             return True
    #         time.sleep(3)

    # def update_expert_status(self, expert):
    #     if expert == "movies": #global version
    #         txn_db = self.db.begin_transaction(read="changes", write="changes")
    #         print("Updating global version")
    #         query = 'FOR doc IN changes UPDATE doc WITH {movies: doc.movies + 1} in changes'
    #         txn_db.aql.execute(query)
    #         txn_db.transaction_status()
    #         txn_db.commit_transaction()
    #         return True
    #     else:
    #         txn_db = self.db.begin_transaction(read="changes", write="changes")
    #         query = 'FOR doc IN changes UPDATE doc WITH {' + expert + ': doc.movies} in changes'
    #         #print(query)
    #         txn_db.aql.execute(query)
    #         txn_db.transaction_status()
    #         txn_db.commit_transaction()
    #         return True


    # ███╗   ██╗███████╗██╗    ██╗    ██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗
    # ████╗  ██║██╔════╝██║    ██║    ██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝
    # ██╔██╗ ██║█████╗  ██║ █╗ ██║    ██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗
    # ██║╚██╗██║██╔══╝  ██║███╗██║    ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝
    # ██║ ╚████║███████╗╚███╔███╔╝    ██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗
    # ╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝     ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝

    def create_pipeline(self, pipeline_id: str, movies_status: list, job_status: dict):
        collection = self.movie_db.db.collection(name = COLLECTION_NAME)
        pipeline_doc = {
            '_key': pipeline_id,
            PIPELINE_MOVIES: movies_status if movies_status else [],
            PIPELINE_TASKS: job_status if job_status else {}
        }
        result = collection.insert(pipeline_doc, overwrite=False)
        return result

    def update_dict(d, **kwargs):
        return {k:update_dict(v, **kwargs) if isinstance(v, dict) else kwargs.get(k,v) for k,v in d.items()}

    def update_pipeline(self, task_name, pipeline_id: str, movies_status: dict, task_status: dict, task_inputs: dict = None):
        db = self.movie_db.db
        # if movies_status is None or len(movies_status) == 0:
        #     error = "Error: no movies_status"
        #     print(error)
        #     return { 'error': error}

        collection = db.collection(name = COLLECTION_NAME)
        error = None
        result = None
        txn_db = db.begin_transaction(read=collection.name, write=collection.name)
        try:
            source_doc = collection.get({'_key': pipeline_id})
            if source_doc:
                tasks = { **source_doc[PIPELINE_TASKS], **task_status } if task_status else source_doc[PIPELINE_TASKS]
                movies = source_doc[PIPELINE_MOVIES]       
                if movies_status:
                    for movie_id, mstatus in movies_status.items():
                        movies[movie_id][TASK_STATUS][task_name] = mstatus[TASK_STATUS][task_name]
                inputs = { **source_doc[PIPELINE_INPUTS], **task_inputs } if task_inputs else source_doc[PIPELINE_INPUTS]
                pipeline_doc = {
                    '_key': pipeline_id,
                    PIPELINE_MOVIES: movies,
                    PIPELINE_TASKS: tasks,
                    PIPELINE_INPUTS: inputs
                }
                result = collection.insert(pipeline_doc, overwrite_mode='update')
            else:
                error = f"Error: no pipeline for {pipeline_id}"

            txn_db.commit_transaction()
        except Exception as e:
            txn_db.abort_transaction()
            error = f"Error: exception {e} in update pipeline for {pipeline_id}"
        return result, error

    def get_pipeline(self, pipeline_id: str, col = None):
        collection = col if col is not None else self.movie_db.db.collection(name = COLLECTION_NAME)
        result = collection.get({'_key': pipeline_id})
        return result

    def get_new_movies(self, pipeline_id: str, task_name: str):
        ''' getting only the list of movies this task hasn't processed or failed '''
        error = None
        result = None
        movie_list, error = self.get_all_movies(pipeline_id, task_name)
        if len(movie_list) > 0:
            result = {}
            for movie_id, movie in movie_list.items():
                if task_name not in movie[TASK_STATUS]:
                    result[movie_id] = movie
                elif not movie[TASK_STATUS][task_name] == STATUS_SUCCESS:
                    result[movie_id] = movie
            # result = filter(lambda movie: ( task_name in movie[TASK_STATUS]), movie_list)
            # result = filter(lambda movie: ( movie[TASK_STATUS][task_name] is not STATUS_SUCCESS), movie_list)
        else:
            error = f"No movies for pipeline for {pipeline_id}"
        return result, error

    def get_all_movies(self, pipeline_id: str, task_name: str):
        ''' getting all the movies for that task'''
        error = None
        result = None
        pipline_doc = self.get_pipeline(pipeline_id=pipeline_id)
        if pipline_doc:
            result = pipline_doc[PIPELINE_MOVIES]
        else:
            error = f"Error: no pipeline for {pipeline_id}"
        return result, error

    def get_first_pipeline_for_task(self, task_name):
        ''' gets the first pipeline for a given task by checking the task name in inputs '''
        # query = f'FOR doc IN {COLLECTION_NAME} FILTER doc.inputs =~ "{task_name}" RETURN doc'
        query = f'FOR doc IN {COLLECTION_NAME} FILTER !HAS(doc.tasks, "{task_name}") RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor:
            return data
        return({})


    def handle_pipeline_task(self,
                             task: PipelineTask,
                             pipeline_id: str,
                             stop_on_failure: bool = False):
        ''' handling pipeline task: get the pipeline movies and process them '''
        status = True
        tstatus = True
        error = None
        task_name = task.get_name()
        new_movies, error = self.get_new_movies(pipeline_id, task_name)
        if len(new_movies):
            movies_status = new_movies.copy()
            for movie_id, movie in movies_status.items():
                status, error = task.process_movie(movie_id) #movie["_id"]
                if not status:
                    tstatus = False
                    if stop_on_failure:
                        return
                # update success status
                movie[TASK_STATUS] = {
                    **movie[TASK_STATUS],
                    task.get_name(): STATUS_FAIL if error else STATUS_SUCCESS
                }
            # update task status
            task_status = {
                task.get_name(): STATUS_SUCCESS if tstatus else STATUS_FAIL
            }
            self.update_pipeline(task.get_name(), pipeline_id, movies_status, task_status)

# {
#     id: jobId from videoprocessing,
#     movies: [
#         { _id: "Movies/308384", _key: "", status: { "videoprocessing": "success/fail", "re-id": "success", ... }},
#         { _id: "Movies/308381", _key: "" }
#     ],
#     tasks: {
#         "videoprocessing": "success/fail",
#         "re-id": "success/fail",
#         ...
#     }
# }

# { _id: "Movies/308384", _key: "", status: { "videoprocessing": { "success": true }, "re-id": { "success": false, "error": "mdf 22 is not found" },  ... }},

## for testing uncomment next line
#__name__ = 'test'
def test():
    pipeline = PipelineApi(None)
    movies_status = [
         { '_id': '123', '_key': '123', 'status': { "videoprocessing": "success" }}
    ]
    jobs = {
        "videoprocessing": "success"
    }
    pipeline.create_pipeline('45f4739b-146a-4ae3-9d06-16dee5df6ca7', movies_status=movies_status, job_status=jobs)
    result = pipeline.get_pipeline('45f4739b-146a-4ae3-9d06-16dee5df6ca7')
    print(result)
    movies_status = [
         { '_id': '234', '_key': '234', 'status': { "videoprocessing": "success" } }
    ]
    # now update
    pipeline.update_pipeline('45f4739b-146a-4ae3-9d06-16dee5df6ca7', movies_status=movies_status, job_status=jobs)

def test_pipeline_task(pipeline_id):
    class MyTask(PipelineTask):
        def process_movie(self, movie_id: str) -> Tuple[bool, str]:
            print (f'handling movie: {movie_id}')
            # task actual work
            return True, None
        def get_name(self) -> str:
            return "my-task"

    pipeline = PipelineApi(None)
    task = MyTask()
    pipeline.handle_pipeline_task(task, pipeline_id, stop_on_failure=True)

if __name__ == 'test':
    pipeline_id='45f4739b-146a-4ae3-9d06-16dee5df6ca7'
    test_pipeline_task(pipeline_id)
    # test()