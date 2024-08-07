class Header:
    def __init__(
        self,
        t_measure_oldest=0,
        t_measure_youngest=0,
        t_measure_average=0,
        num_measurements=1,
    ):
        self.sender_id = None
        self.destination_id = None
        self.t_measure_oldest = t_measure_oldest
        self.t_measure_youngest = t_measure_youngest
        self.t_measure_average = t_measure_average
        self.num_measurements = num_measurements


class Message(object):
    def __init__(self, header: Header, data: object = None):
        self.header = header
        self.data = data
