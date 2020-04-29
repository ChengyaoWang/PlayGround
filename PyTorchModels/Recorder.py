import json

class Recorder(object):

    def __init__(self):
        self.Record = {}

    def add_record_(self, key, dict):
        self.Record[key] = dict

    # Save the config into JSON
    def json_dump_(self, PATH):
        with open(PATH + '.json', 'w') as f:
            json.dump(self.Record, f)
