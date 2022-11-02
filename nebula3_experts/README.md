# nebula3_experts
## Base module for all experts

## Expert Configuration
- Since experts are started in a microservice container (via gradient or else), it is useless to
  set configuration in the command args, therefore, expert's specific configuration
  (model type, model tune params etc) should come from env.
  Further info can be found in: https://12factor.net/config
- every expert can have something similar to ExpertsConf class like NEBULA_CONF

## CLI
- CLI is now done via the REST api, each expert microservice app has a web server with
  all the apis documented via the /docs (or /redoc) url, which also provide a basic way
  to operate the experts api.
- Postman and other apps can be used to operate the apis
- If you still want to work from terminal, learn curl.

#
- when an expert starts a new taks it has to call self.add_task and when finishes call self.remove_task
- since expert is running in container, all the logs are going to stdout/stderr so that they could be
  seen from outside using docker logs, and that we can aggragate them


# Env params
- EXPERT_RUN_PIPELINE: true/false run pipeline thread (default - false)

# From Notebook to Microservice
- Divide your code to
  -  **init phase**: setup, load model, download pth, pkl etc'
  - **run phase**: getting params for predicting: movie file, frames, mdfs, scenes, batch size, activating the model: predict/forward and transforming results to a list of TokenRecord
- All the cfg params for the init the phase will be defined using the environment variables
- All the cfg paramd for run phase are passed in the REST call (/predict) in the json body
- Create a class [ExpertName]Expert e.g: VgExpert which inherits from BaseExpert.
- Implement at least the following methods:
  - `__init__(self)`: **init phase** code goes here
  - `get_name`: return the name of the expert, this name is also used in the Tokens Record in DB or as returned json
  - `predict(self, expert_params: ExpertParam)`: **run phase** code goes here
- in the `__main__` section add:

 `

    vg_expert = VgExpert()
    expert_app = ExpertApp(expert=vg_expert)
    app = expert_app.get_app()
    expert_app.run()
- Basic example can be found in main.py under test folder

- Since the expert run as a module inside an Application Server called FastAPI (which uses uvicorn) you run the uvicorn and direct it to the expert module (assuming vg_expert.py is under vg folder):
  - `uvicorn vg.vg_expert:app --host 0.0.0.0`
  -  default port: 8000
- Prepare an environment.yml file with all your packages in your current conda env (conda env export > environment.yml)

### Building a Microservice Docker image
- Build a docker image: basic Dockerfiles and info (*README.md*) is under Docker/microservice folder
- Build the docker with a Dockerfile from Docker/microservice folder (note: different files for cpu/gpu), ensure that you have the correct environment.yml in your top folder
- run it with either command or with docker-compose:
  - Command: docker run -it -p 8000:8000 -e "ARANGO_HOST=172.83.9.249" -n my_expert docker/image-name
  - Compose: check the docker.compose.yml file in Docker/microservice folder
- push the image to your docker hub registry (try to add versioning in the tag name e.g: the commit hash prefix)

## Deploy your microservice

- Go to deployments in Gradient and follow the instructions
- for deployments.yaml you can use the one located at Docker/microservice folder and change the image name to your image

## Predict list of images
- you can post a request to handle multiple images in one request, there are 2 modes of operation: sync where you'll wait for the answer (can take sometime) or async where you'll get a job-id and can query by job id and get the results when ready.
- Async: create a POST method:
  ```
  POST /predict/images
  {
    "images": [
        { "image_id": "2402585", "url": "https://....jpg" },
        { "image_id": "2402585", "url": "https://....jpg" }
    ],
    "output" : "json",
    "is_async": true,
    "extra_params": {
    }
  }
- you'll receive a jobId, you can then query the job status using this request:
  ```
  GET http://localhost:8000/jobs/[job-id]
- or get all the jobs by sending:
  ```
  GET http://localhost:8000/jobs
- the job results are cached 10 minutes (can be changed with env param: JOBS_EXPIRATION)

# todo:
- add get/set logger level
- async operations


## Package Versions:
- 1.2.1: add get get_first_pipeline_for_task and 'task' mode: EXPERT_RUN_MODE='task' ( in env) ,
        added testing in test/main.py
- 1.2.0: start with pipeline api and tasks
- 1.1.0:
  - add predict list of mages (/predict/images)
  - add async operation with job_id
  - add get jobs (/jobs) or specific job by id (/jobs/[job_id])
- 1.0.1 change db package to 0.2.2
- 1.0.0 initial version with batch mode