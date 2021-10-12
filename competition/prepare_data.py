import os
import logging
import argparse
from datetime import datetime

from competition.warehouse import Engine


def get_dwh() -> Engine:
    engine = Engine()
    
    engine.register_table('train', 'input/train.csv')
    engine.register_table('train_raw', 'input/train_raw.csv')
    engine.register_table('sample_test', 'input/sample_test.csv')
    # engine.register_table('city_town_village', 'input/city_town_village.geojson')
    # engine.register_table('russia_latest_osm', 'input/russia-latest.osm.pbf')

    # era_data_path = 'input/ERA5_data'
    # for fn in os.listdir(era_data_path):
    #     engine.register_table(fn.split('.')[0],
    #                           os.path.join(era_data_path, fn))

    return engine
