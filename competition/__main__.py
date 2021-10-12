if __name__ == '__main__':
    import cfgrib

    ds = cfgrib.open_datasets('input/ERA5_data/temp_2018.csv')

    print(ds)