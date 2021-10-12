import os

from .engine import Engine


def get_dwh(data_path: str) -> Engine:
    engine = Engine()
    
    engine.register_table('train', os.path.join(data_path, 'train.csv'))
    engine.register_table('train_raw', os.path.join(data_path, 'train_raw.csv'))
    engine.register_table('sample_test', os.path.join(data_path, 'sample_test.csv'))
    # engine.register_table('city_town_village', os.path.join(data_path, 'city_town_village.geojson'))
    # engine.register_table('russia_latest_osm', os.path.join(data_path, 'russia-latest.osm.pbf'))

    # era_data_path = os.path.join(data_path, 'ERA5_data')
    # for fn in os.listdir(era_data_path):
    #     engine.register_table(fn.split('.')[0],
    #                           os.path.join(era_data_path, fn))

    return engine