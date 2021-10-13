import os
import logging
import argparse
from datetime import datetime

from competition.warehouse import Engine, get_dwh
from competition.featurise import Featuriser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="warning", )
    args = vars(parser.parse_args())

    logging.basicConfig(filename=f'logs/{str(datetime.now())}.log',
                        level=getattr(logging, args['log'].upper(), None))

    # prepare_data
    engine = get_dwh('input')

    # compute features
    train_df = Featuriser(engine, 'train', 'target_base').get_features({'dates_features': {}})
    print(train_df.head())
    test_df = Featuriser(engine, 'sample_test', ).get_features({'dates_features': {}})
    print(test_df.head())
    # fit pipeline
    # submit
