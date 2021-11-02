import os
import sys
import warnings
import logging

import yaml
import pandas as pd

from competition.repository import Repository
from competition.warehouse import Warehouse
from competition.featurise import Featuriser


warnings.simplefilter("ignore")
logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%Y-%m-%d,%H:%M:%S",
                    level=logging.INFO, )
logger = logging.getLogger(__name__)


def featurise(config: dict, repository: Repository) -> None:
    engine = (Warehouse(data_path=config['warehouse']['data_path'], train=True)
              .create(config['warehouse']['create_args']))
    
    features_df = (Featuriser(engine, repository)
                   .get_features(config['featurise']['calcers']))
    
    features_df.to_csv('features.csv', index=False)


def train(config: dict, repository: Repository) -> None:
    features_df = pd.read_csv('features.csv')

    pipeline = (repository
                .get_object(config['pipeline']['name'])(config['pipeline']['seed']))
    
    pipeline.fit(features_df, config['pipeline'], repository)
    pipeline.save(os.path.join('models', config['name']))



def submit(config: dict, repository: Repository) -> None:
    name = config['name']

    calcers_config = config['featurise']['calcers']
    del calcers_config['target_base']
    config['featurise']['calcers'] = calcers_config

    with open('solution.yaml', 'w') as f:
        yaml.dump(config, f)

    if os.path.exists('submits/{name}'):
        os.system(f"rm -dR submits/{name} ")

    os.system(f"mkdir -p submits/{name} "
              f"&& mkdir -p submits/{name}/models "
              f"&& cp solution.py submits/{name}/solution.py "
              f"&& cp metadata.json submits/{name}/metadata.json "
              f"&& cp solution.yaml submits/{name}/solution.yaml "
              f"&& cp -R competition submits/{name}/competition "
              f"&& cp -a models/{name}/* submits/{name}/models/ "
              f"&& cd submits/{name} "
              f"&& zip -r {name}_submission.zip * "
              f"&& mv {name}_submission.zip ../{name}_submission.zip "
              "&& cd .."
              f"&& rm -dR {name} ")


tasks = {'featurise': featurise,
         'train': train, 
         'submit': submit, }


if __name__ == '__main__':
    task, config_name = sys.argv[1:]
    with open(os.path.join('configs/', config_name + '.yaml'), 'r') as f:
        config = yaml.load(f, yaml.Loader)

    repository = Repository()
    for name, obj_name in config['repository'].items():
        repository.register(name, obj_name)

    tasks[task](config, repository)
