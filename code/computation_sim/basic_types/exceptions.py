class ComputationSimError(Exception):
    pass


class CommunicationError(ComputationSimError):
    pass


class BadNodeGraphError(ComputationSimError):
    pass
