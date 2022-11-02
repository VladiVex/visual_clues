[pipeline]
- each service (except videoprocessing) is activated with env param of "jobId"
  identifying the job is it about to execute
  in the DB there's a collection called pipeline
  pipeline collection:
  {
    id: jobId from videoprocessing,
    movies: [
        { _id: , _key: , status: { "videoprocessing": "success/fail", ... }},
        { _id: , _key: }
    ],
    jobs: {
        "videoprocessing": "success/fail",
        ...
    }
  }

- Pipeline api additions:
  - create pipeline: create_pipeline(pipeline_id, movie status list, job_status) - pipeline_id is the jobId
  - update pipeline status: update_pipeline(pipeline_id, movie status list, job status)
  - get pipline: get_pipeline(pipeline_id)

[paperspace-workflow]
- we can install gradient sdk and create and run a workflow, the workflow-id can be the generated
  job-id, also we can create a secret for that

[proposed pipeline record]
  {
    id: workflow-id // the workflow-id used to run the workflow in gradient
    movies: [],
    tasks: { },
    inputs: {
      "videoprocessing": {
        "movies": [
            { "movie_id": "", "url": "http://74.82.29.209:9000/msrvtt/video8059.mp4", "type": "file" },
            { "movie_id": "2402585", "url": "http://74.82.29.209:9000/msrvtt/video8135.mp4" , "type": "file" }
        ],
        "save_movies": true,
        "output" : "db",
        "is_async": true,
        "overwrite": true
      }
    }
  }
- Running in batch/task mode: don't run uvicorn, run regular 'python [filename.py]'