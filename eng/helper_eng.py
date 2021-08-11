def load_lang_data_eng():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eng/location.csv")
    frameworks_location = pd.read_csv(data_path, delimiter=', ', engine='python', header=None).values
    frameworks_location = dict(zip(frameworks_location[:, 0], frameworks_location[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eng/orientation.csv")
    frameworks_orientation = pd.read_csv(data_path, delimiter=', ', engine='python', header=None).values
    frameworks_orientation = dict(zip(frameworks_orientation[:, 0], frameworks_orientation[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eng/yolov3.csv")
    data_multilingual_obj_names = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    return frameworks_location, frameworks_orientation, data_multilingual_obj_names