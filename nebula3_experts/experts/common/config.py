import os

class ExpertConf:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ExpertConf, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.RUN_PIPELINE = eval(os.getenv('EXPERT_RUN_PIPELINE', 'False'))
        self.RUN_MODE = os.getenv('EXPERT_RUN_MODE', 'service')
        self.THREAD_POOL_SIZE = int(os.getenv('EXPERT_THREAD_POOL_SIZE', '1'))
        self.JOBS_EXPIRATION = int(os.getenv('JOBS_EXPIRATION', '600')) # 10 MINUTES
        self.MAX_WAIT_PREDICT_TIME = int(os.getenv('MAX_WAIT_PREDICT_TIME', '600')) # 10 MINUTES

    def get_run_pipeline(self):
        return (self.RUN_PIPELINE)

    def get_run_mode(self):
        return (self.RUN_MODE)
    def get_thread_pool_size(self):
        return (self.THREAD_POOL_SIZE)
    def get_jobs_expiration(self):
        return (self.JOBS_EXPIRATION)
    def get_max_wait_predict_time(self):
        return (self.MAX_WAIT_PREDICT_TIME)
