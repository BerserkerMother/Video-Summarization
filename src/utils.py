import yaml
import json


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.num = 0

    def update(self, val, num):
        self.val += val*num
        self.num += num

    def avg(self):
        return self.val / self.num


def load_yaml(path):
    with open(path, 'r') as f:
        try:
            splits = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    return splits


def load_json(path):
    with open(path) as f:
        splits = json.load(f)

    return splits
