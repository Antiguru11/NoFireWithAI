import os
import pathlib
import logging
import warnings

import yaml

from competition.repository import Repository
from competition.warehouse import Warehouse
from competition.featurise import Featuriser


warnings.simplefilter("ignore")
logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%Y-%m-%d,%H:%M:%S",
                    level=logging.INFO, )
logger = logging.getLogger(__name__)


base_path = pathlib.Path(__file__).parent
input_path = base_path / "input/"
models_path = base_path / "models/"
output_path = base_path / "output/"
add_data_path = base_path / "additional_data/"


if __name__ == "__main__":
    with open('solution.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)
    
    repository = Repository()
    for name, type_name in config['repository'].items():
        repository.register(name, type_name)

    engine = (Warehouse(input_path, train=False)
              .create(config['warehouse']['create_args']))

    features_df = (Featuriser(engine, repository)
                   .get_features(config['featurise']['calcers']))

    pipeline = (repository
                .get_object(config['pipeline']['name'])
                .load(models_path, config['pipeline']['seed']))

    predicts_df = pipeline.predict(features_df)
    predicts_df.to_csv(os.path.join(output_path, "output.csv"),
                       index_label="id", )
