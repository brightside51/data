import sys

def get_ds(
    data_args = None,
    mode = 'train',
):
    sys.path.append(f"{data_args.reader_fp}")
    print(data_args.reader_fp)

    # Duke Breast Cancer Dataset
    if data_args.dataset == 'dukebreast':
        from dukebreast_reader import NCDataset as DukebreastDataset
        ds = DukebreastDataset(data_args, mode = mode)
    
    # MetaBreast Cancer Dataset
    elif data_args.dataset == 'metabreast':
        from metabreast_reader import NCDataset as MetabreastDataset
        ds = MetabreastDataset(data_args, mode = mode)

    # LIDC-IDRI Dataset
    elif data_args.dataset == 'lidc_idri':
        from lidc_idri_reader import DataReader as LIDCIDRIDataset
        ds = LIDCIDRIDataset(data_args, mode = mode)

    # LUNA25 Nodule Dataset
    elif data_args.dataset == 'luna25_nodule':
        from luna25_nodule_reader import NCDataset as LUNA25NoduleDataset
        ds = LUNA25NoduleDataset(data_args, mode = mode)

    elif data_args.dataset == 'covid_jun2020':
        from covid_jun2020_reader import DataReader as CovidJun2020Dataset
        ds = CovidJun2020Dataset(data_args, mode = mode)



    elif data_args.dataset == 'ds003768':
            from ds003768_reader import DataReader as ds003768Dataset
            ds = ds003768Dataset(data_args, mode = mode)

    elif data_args.dataset == 'ds007216':
        from ds007216_reader import DataReader as ds007216Dataset
        ds = ds007216Dataset(data_args, mode = mode)

    elif data_args.dataset == 'noddi':
        from noddi_reader import DataReader as NODDIDataset
        ds = NODDIDataset(data_args, mode = mode)

    elif data_args.dataset == 'natview':
        from natview_reader import DataReader as NatViewDataset
        ds = NatViewDataset(data_args, mode = mode)

    elif data_args.dataset == 'ds002336':
        from ds002336_reader import DataReader as ds002336Dataset
        ds = ds002336Dataset(data_args, mode = mode)

    elif data_args.dataset == 'ds001116':
        from ds001116_reader import DataReader as ds001116Dataset
        ds = ds001116Dataset(data_args, mode = mode)

    elif data_args.dataset == 'ds002158':
        from ds002158_reader import DataReader as ds002158Dataset
        ds = ds002158Dataset(data_args, mode = mode)

    return ds
