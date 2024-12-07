from utils.argUtils import CustomObject, get_yaml_loader
from fineTuning import train
from evaluation import evaluate
from retriever.indexer import create_index
from retriever.search import run_retrieval
from retriever.analyse import score_results
import yaml
import json

def start(configPath):
    with open(configPath, 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

    if Args.FineTuning.Action:
        train(Args)

    if Args.Evaluate.Action:
        evaluate(Args)

    if Args.Retrieval.Action:
        create_index(Args)
        run_retrieval(Args)
        score_results(Args)


if __name__ == '__main__':
    start('configEval.yaml')

