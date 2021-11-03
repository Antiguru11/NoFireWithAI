#!/bin/bash

rm -R -f input
mkdir -p input/ERA5_data
wget -O input/train_raw.csv https://dsworks.s3pd01.sbercloud.ru/aij2021/NoFireWithAI/train_raw.csv
wget -O input/sample_test.csv https://dsworks.s3pd01.sbercloud.ru/aij2021/NoFireWithAI/sample_test.csv
wget -O input/train.csv https://dsworks.s3pd01.sbercloud.ru/aij2021/NoFireWithAI/train.csv
wget -O input/russia-latest.osm.pbf https://dsworks.s3pd01.sbercloud.ru/aij2021/NoFireWithAI/russia-latest.osm.pbf
wget -O input/city_town_village.geojson https://dsworks.s3pd01.sbercloud.ru/aij2021/NoFireWithAI/city_town_village.geojson

for year in 2018 2019 2020 2021
do
    for file in temp wind vegetation heat1 heat2 evaporation1 evaporation2
    do
        wget -O input/ERA5_data/${file}_${year}.grib https://dsworks.s3pd01.sbercloud.ru/aij2021/NoFireWithAI/${file}_${year}.grib
    done
done