import sys

def get_ds(
    data_args = None,
    mode = 'train',
):
    sys.path.append(f"{data_args.reader_fp}")

    # Duke Breast Cancer Dataset
    if data_args.dataset == 'dukebreast':
        from dukebreast_reader import NCDataset as DukebreastDataset
        ds = DukebreastDataset(data_args, mode = mode)
    
    # MetaBreast Cancer Dataset
    elif data_args.dataset == 'metabreast':
        from metabreast_reader import NCDataset as MetabreastDataset
        ds = MetabreastDataset(data_args, mode = mode)
    
    return ds
