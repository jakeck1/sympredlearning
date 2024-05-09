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



def save_circular_rw_data(walks,SRs,path,makepath=True,overwrite=False,suffices=None):

    if makepath:
        if not os.path.exists(path):
            os.makedirs(path)

    if isinstance(suffices,str):
        suffices = list(suffices)





    file_name = os.path.join(path,'walks')
    if suffices is not None:
        for suffix in suffices:
            file_name += '_' + suffix
    file_name +='.npz'


    if (not overwrite) and (os.path.isfile(file_name)):
        print('file at {} already exists, set overwrite=True if you want to create data anyway'.format(file_name))

    else:
        np.savez(file_name,walks)
        print('saved data at {}'.format(file_name))



    file_name = os.path.join(path,'SRs')
    if suffices is not None:
        for suffix in suffices:
            file_name += '_' + suffix
    file_name +='.npz'


    if (not overwrite) and (os.path.isfile(file_name)):
        print('file at {} already exists, set overwrite=True if you want to create data anyway'.format(file_name))

    else:
        np.savez(file_name,SRs)
        print('saved data at {}'.format(file_name))


def load_circular_rw_data(path,suffices):

    file_name = os.path.join(path, 'walks')
    if suffices is not None:
        for suffix in suffices:
            file_name += '_' + suffix
    file_name += '.npz'

    walks = np.load(file_name)



    file_name = os.path.join(path, 'SRs')
    if suffices is not None:
        for suffix in suffices:
            file_name += '_' + suffix
    file_name += '.npz'

    SRs = np.load(file_name)

    return walks,SRs