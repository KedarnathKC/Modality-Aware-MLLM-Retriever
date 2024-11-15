from utils.argUtils import CustomObject, get_yaml_loader
from fineTuning import train
from evaluation import evaluate
import yaml
import json
loader = yaml.SafeLoader


def start(configPath):
    with open(configPath, 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

    if Args.FineTuning.Action:
        train(Args)

    if Args.Evaluate.Action:
        evaluate(Args)

if __name__ == '__main__':
    start('config.yaml')

