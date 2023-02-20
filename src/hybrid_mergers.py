class Merger_Hybrid:
    """
    1. Combines multiple strategies, allowing different strategies at different rounds.
    """
    def __init__(self, mergers, indices):
        self.mergers = mergers
        self.indices = indices

    def __call__(self, outputs):
        return self.mergers[self.indices[outputs[0].round]](outputs)

    def reset(self):
        for merger in self.mergers: merger.reset()
        return self