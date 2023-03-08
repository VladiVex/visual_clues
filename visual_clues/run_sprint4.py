from experts.pipeline.api import PipelineApi, PipelineTask
from run_visual_clues import TokensPipeline
import os
from typing import Tuple

def test_pipeline_task(pipeline_id):
    class MyTask(PipelineTask):
        def __init__(self):
            super().__init__(pipeline_id)
            self.visual_clues_pipeline = TokensPipeline()
            print("Initialized successfully.")

        def process_movie(self, movie_id: str) -> Tuple[bool, str]:
            print (f'handling movie: {movie_id}')

            output = self.visual_clues_pipeline.run_visual_clues_pipeline(movie_id)

            print("Finished handling movie.")
            print(output)
            return output
        def get_name(self) -> str:
            return "visual_clues"

    pipeline = PipelineApi(None)
    task = MyTask()
    pipeline.handle_pipeline_task(task, pipeline_id, stop_on_failure=True)

def test():
    pipeline_id = os.environ.get('PIPELINE_ID')
    # print(pipeline_id)
    # pipeline_id='12345678'
    test_pipeline_task(pipeline_id)

if __name__ == '__main__':
    test()
