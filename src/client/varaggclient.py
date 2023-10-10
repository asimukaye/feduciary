from .baseclient import  BaseClient

class VaraggClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)