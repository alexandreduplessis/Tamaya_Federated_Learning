class ClientOutput:
    def __init__(self, client_id):
        self.client_id = client_id
        self.size = None
        self.weight = None
        self.round = 0
        self.losses = []