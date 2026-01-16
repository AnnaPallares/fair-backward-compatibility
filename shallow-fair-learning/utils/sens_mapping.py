def sens_loc(dataset_name):
    # Mapping of the sensitive var for the datasets. It may need to be re-checked by the user.
    mapping = {'adult': -1, 'compas': -1, 'german': -1, 'arrhythmia': -1, 'bank-additional': 21, 'student-por': 14} 
    return mapping.get(dataset_name.lower(), 0)


