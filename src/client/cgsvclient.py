from .baseclient import  BaseClient

class CgsvClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)