import numpy as np
import os




def save_convergence_data(data,path,weightnames = None,suffices=None,makepath=True,overwrite=False):

    if weightnames is None:
        weightnames = ['W','V']

    assert len(data)==len(weightnames), 'number of weight labels does not match number of arrays to save'

    if makepath:
        if not os.path.exists(path):
            os.makedirs(path)

    if isinstance(suffices,str):
        suffices = list(suffices)


    for i in range(len(data)):


        file_name = os.path.join(path,'convergence_{}'.format(weightnames[i]))
        if suffices is not None:
            for suffix in suffices:
                file_name += '_' + suffix
        file_name +='.npy'


        if (not overwrite) and (os.path.isfile(file_name)):
            print('file at {} already exists, set overwrite=True if you want to create data anyway'.format(file_name))

        else:
            np.save(file_name,data[i])
            print('saved data at {}'.format(file_name))


def load_convergence_data(path, weightnames=None, suffices=None):
    if weightnames is None:
        weightnames = ['W', 'V']

    data = []
    for i in range(len(weightnames)):

        file_name = os.path.join(path, 'convergence_{}'.format(weightnames[i]))
        if suffices is not None:
            for suffix in suffices:
                file_name += '_' + suffix
        file_name += '.npy'
        arr = np.load(file_name)
        data.append(arr)
        print('loaded data from {}'.format(file_name))
    return data



def save_shift_data(data,path,weightnames = None,suffices=None,makepath=True,overwrite=False):

    if weightnames is None:
        weightnames = ['ca3','ca1']

    assert len(data)==len(weightnames), 'number of weight labels does not match number of arrays to save'

    if makepath:
        if not os.path.exists(path):
            os.makedirs(path)


    if isinstance(suffices,str):
        suffices = list(suffices)


    for i in range(len(data)):


        file_name = os.path.join(path,'shifts_{}'.format(weightnames[i]))
        if suffices is not None:
            for suffix in suffices:
                file_name += '_' + suffix
        file_name +='.npy'


        if (not overwrite) and (os.path.isfile(file_name)):
            print('file at {} already exists, set overwrite=True if you want to create data anyway'.format(file_name))

        else:
            np.save(file_name,data[i])
            print('saved data at {}'.format(file_name))

def load_shift_data(path, weightnames=None, suffices=None):
    if weightnames is None:
        weightnames = ['ca3', 'ca1']

    data = []
    for i in range(len(weightnames)):

        file_name = os.path.join(path, 'shifts_{}'.format(weightnames[i]))
        if suffices is not None:
            for suffix in suffices:
                file_name += '_' + suffix
        file_name += '.npy'
        arr = np.load(file_name)
        data.append(arr)
        print('loaded data from {}'.format(file_name))
    return data



def save_circular_rw_data(walks, SRs, path, makepath=True, overwrite=False, suffices=None):
    if makepath and not os.path.exists(path):
        os.makedirs(path)

    if isinstance(suffices, str):
        suffices = [suffices]

    # Function to construct the file name with suffices
    def construct_file_name(base_name):
        file_name = os.path.join(path, base_name)
        if suffices:
            for suffix in suffices:
                file_name += '_' + suffix
        return file_name + '.npz'

    # Save walks
    walks_file_name = construct_file_name('walks')
    if (not overwrite) and os.path.isfile(walks_file_name):
        print(f'File at {walks_file_name} already exists. Set overwrite=True if you want to create data anyway.')
    else:
        np.savez(walks_file_name, **walks)
        print(f'Saved data at {walks_file_name}')

    # Save SRs
    SRs_file_name = construct_file_name('SRs')
    if (not overwrite) and os.path.isfile(SRs_file_name):
        print(f'File at {SRs_file_name} already exists. Set overwrite=True if you want to create data anyway.')
    else:
        np.savez(SRs_file_name, **SRs)
        print(f'Saved data at {SRs_file_name}')

def load_circular_rw_data(path, suffices=None):
    # Function to construct the file name with suffices
    def construct_file_name(base_name):
        file_name = os.path.join(path, base_name)
        if suffices:
            for suffix in suffices:
                file_name += '_' + suffix
        return file_name + '.npz'

    # Load walks
    walks_file_name = construct_file_name('walks')
    walks_data = np.load(walks_file_name, allow_pickle=True)
    walks = {key: walks_data[key] for key in walks_data.files}

    # Load SRs
    SRs_file_name = construct_file_name('SRs')
    SRs_data = np.load(SRs_file_name, allow_pickle=True)
    SRs = {key: SRs_data[key] for key in SRs_data.files}

    return walks, SRs
