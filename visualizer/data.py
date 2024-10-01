import os
from csv import reader

class SisFall():
    def __init__(self, subject_id, code=None):
        self.subject_id = subject_id
        self.code = code

        self.dir_name = os.path.join(os.path.dirname(__file__), 'SisFall_dataset', subject_id)
        self.file_dir = os.listdir(self.dir_name)
        self.file_dir = list(filter(lambda x: subject_id in x, self.file_dir))
        if code:
            self.file_dir = list(filter(lambda x: x.startswith(code), self.file_dir))
    
    def read(self):
        out = {}
        for file in self.file_dir:
            with open(os.path.join(self.dir_name, file)) as content:
                csv_reader = reader(content)
                out[file] = list(csv_reader)
        return out