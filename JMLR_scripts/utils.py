import math
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR
from scipy import ndimage


def getData(dataset_name, train_size, seed):

    if dataset_name == 'MNIST_rot':
        return getMNISTROT(train_size, seed)
    elif dataset_name == 'MNIST_rot_noaug':
        return getMNISTROTNOAUG(train_size, seed)
    elif dataset_name == 'MNIST_rot_back':
        return getMNISTROTBACK(train_size, seed)
    elif dataset_name == 'MNIST_rot_back_noaug':
        return getMNISTROTBACKNOAUG(train_size, seed)
    elif dataset_name == 'MNIST':
        return getMNIST(train_size, seed)
    elif dataset_name == 'MNIST_back':
        return getMNISTBACK(train_size, seed)
    elif dataset_name == 'Galaxy':
        return getGalaxy(train_size, seed)
    elif dataset_name == 'Galaxy_noaug':
        return getGalaxyNOAUG(train_size, seed)
    elif dataset_name == 'Malaria':
        return getMalaria(train_size, seed)
    elif dataset_name == 'Malaria_noaug':
        return getMalariaNOAUG(train_size, seed)
    elif dataset_name == 'bigearthnet':
        return getBigEarthNet(train_size, seed)
    elif dataset_name == 'bigearthnet_noaug':
        return getBigEarthNetNOAUG(train_size, seed)
    else:
        raise Exception("Dataset not implemented")


def getMNISTROT(train_size, seed):

    # Loading data
    images_folder_train = 'C:/Users/vdelchev/Documents/datasets/MNIST_rot/train.amat'
    images_folder_test = 'C:/Users/vdelchev/Documents/datasets/MNIST_rot/test.amat'

    #images_folder_train = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST_rot/train.amat'
    #images_folder_test = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST_rot/test.amat'

    train = np.loadtxt(images_folder_train)
    test = np.loadtxt(images_folder_test)

    all_data = np.concatenate((train, test), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(all_data[:,0:784], all_data[:,-1], 
                                                        test_size=1.-train_size, 
                                                        random_state=seed, stratify=all_data[:,-1])

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    class RotMNISTDataset(Dataset):
        '''Rot-MNIST Dataset'''

        def __init__(self, image_set, label_set, transform=None, rot=False):
            '''
            Args:
                image_set (numpy int): matrix containing images of 
                rot-MIST digits
                label_set (numpy int): matrix containing labels for
                the digits
            '''
            self.image_set = image_set
            self.label_set = label_set
            self.transform = transform
            self.rot = rot

        def __len__(self):
            '''
            Returns length of image_set
            '''
            return len(self.image_set)

        def __getitem__(self, idx):
            '''
            Behavior: Takes a random index from the instance of Dataloader and returns
            the respective sample from the data
            Args:
                idx (int): denotes index of sample to be returned
            Returns:
                image_sample (torch tensor): 1D matrix containing the image sample
                label_sample (torch tensor): label of the respective sample
            '''

            image_sample = self.image_set[idx].astype(np.float32)
            label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

            if self.transform is not None:
                if self.rot:
                    angle_in_degrees = np.random.rand() * 360.
                    image_sample = ndimage.rotate(image_sample, angle_in_degrees, reshape=False, mode='reflect', order=2)
                image_sample = self.transform(image_sample)

            return image_sample, label_sample

    trans_train = transforms.Compose([transforms.ToTensor()])

    trans_valid = transforms.Compose([transforms.ToTensor()])

    # creating train_loader and valid_loader
    train_dataset = RotMNISTDataset(X_train, y_train, transform=trans_train, rot=True)
    valid_dataset = RotMNISTDataset(X_test, y_test, transform=trans_valid, rot=False)
    print(len(train_dataset), len(valid_dataset))

    batch_size = 64

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=0)

    return trainloader, validloader


def getMNISTROTNOAUG(train_size, seed):

    # Loading data
    #images_folder_train = 'C:/Users/vdelchev/Documents/datasets/MNIST_rot/train.amat'
    #images_folder_test = 'C:/Users/vdelchev/Documents/datasets/MNIST_rot/test.amat'

    images_folder_train = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST_rot/train.amat'
    images_folder_test = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST_rot/test.amat'

    train = np.loadtxt(images_folder_train)
    test = np.loadtxt(images_folder_test)

    all_data = np.concatenate((train, test), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(all_data[:,0:784], all_data[:,-1], 
                                                        test_size=1.-train_size, 
                                                        random_state=seed, stratify=all_data[:,-1])

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    class RotMNISTDataset(Dataset):
        '''Rot-MNIST Dataset'''

        def __init__(self, image_set, label_set, transform=None, rot=False):
            '''
            Args:
                image_set (numpy int): matrix containing images of 
                rot-MIST digits
                label_set (numpy int): matrix containing labels for
                the digits
            '''
            self.image_set = image_set
            self.label_set = label_set
            self.transform = transform
            self.rot = rot

        def __len__(self):
            '''
            Returns length of image_set
            '''
            return len(self.image_set)

        def __getitem__(self, idx):
            '''
            Behavior: Takes a random index from the instance of Dataloader and returns
            the respective sample from the data
            Args:
                idx (int): denotes index of sample to be returned
            Returns:
                image_sample (torch tensor): 1D matrix containing the image sample
                label_sample (torch tensor): label of the respective sample
            '''

            image_sample = self.image_set[idx].astype(np.float32)
            label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

            if self.transform is not None:
                if self.rot:
                    angle_in_degrees = np.random.rand() * 360.
                    image_sample = ndimage.rotate(image_sample, angle_in_degrees, reshape=False, mode='reflect', order=2)
                image_sample = self.transform(image_sample)

            return image_sample, label_sample

    trans_train = transforms.Compose([transforms.ToTensor()])

    trans_valid = transforms.Compose([transforms.ToTensor()])

    # creating train_loader and valid_loader
    train_dataset = RotMNISTDataset(X_train, y_train, transform=trans_train, rot=False)
    valid_dataset = RotMNISTDataset(X_test, y_test, transform=trans_valid, rot=False)
    print(len(train_dataset), len(valid_dataset))

    batch_size = 64

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=4)

    return trainloader, validloader


def getMNISTROTBACK(train_size, seed):

    # Loading data
    #images_folder_train = 'C:/Users/vdelchev/Documents/datasets/MNIST_rot_back/train.amat'
    #images_folder_test = 'C:/Users/vdelchev/Documents/datasets/MNIST_rot_back/test.amat'

    images_folder_train = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST_rot_back/train.amat'
    images_folder_test = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST_rot_back/test.amat'

    train = np.loadtxt(images_folder_train)
    test = np.loadtxt(images_folder_test)

    all_data = np.concatenate((train, test), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(all_data[:,0:784], all_data[:,-1], 
                                                        test_size=1.-train_size, 
                                                        random_state=seed, stratify=all_data[:,-1])

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    class RotMNISTDataset(Dataset):
        '''Rot-MNIST Dataset'''

        def __init__(self, image_set, label_set, transform=None, rot=False):
            '''
            Args:
                image_set (numpy int): matrix containing images of 
                rot-MIST digits
                label_set (numpy int): matrix containing labels for
                the digits
            '''
            self.image_set = image_set
            self.label_set = label_set
            self.transform = transform
            self.rot = rot

        def __len__(self):
            '''
            Returns length of image_set
            '''
            return len(self.image_set)

        def __getitem__(self, idx):
            '''
            Behavior: Takes a random index from the instance of Dataloader and returns
            the respective sample from the data
            Args:
                idx (int): denotes index of sample to be returned
            Returns:
                image_sample (torch tensor): 1D matrix containing the image sample
                label_sample (torch tensor): label of the respective sample
            '''

            image_sample = self.image_set[idx].astype(np.float32)
            label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

            if self.transform is not None:
                if self.rot:
                    angle_in_degrees = np.random.rand() * 360.
                    image_sample = ndimage.rotate(image_sample, angle_in_degrees, reshape=False, mode='reflect', order=2)
                image_sample = self.transform(image_sample)

            return image_sample, label_sample

    trans_train = transforms.Compose([transforms.ToTensor()])

    trans_valid = transforms.Compose([transforms.ToTensor()])

    # creating train_loader and valid_loader
    train_dataset = RotMNISTDataset(X_train, y_train, transform=trans_train, rot=True)
    valid_dataset = RotMNISTDataset(X_test, y_test, transform=trans_valid, rot=False)
    print(len(train_dataset), len(valid_dataset))

    batch_size = 64

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=4)

    return trainloader, validloader


def getMNISTROTBACKNOAUG(train_size, seed):

    # Loading data
    #images_folder_train = 'C:/Users/vdelchev/Documents/datasets/MNIST_rot_back/train.amat'
    #images_folder_test = 'C:/Users/vdelchev/Documents/datasets/MNIST_rot_back/test.amat'

    images_folder_train = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST_rot_back/train.amat'
    images_folder_test = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST_rot_back/test.amat'

    train = np.loadtxt(images_folder_train)
    test = np.loadtxt(images_folder_test)

    all_data = np.concatenate((train, test), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(all_data[:,0:784], all_data[:,-1], 
                                                        test_size=1.-train_size, 
                                                        random_state=seed, stratify=all_data[:,-1])

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    class RotMNISTDataset(Dataset):
        '''Rot-MNIST Dataset'''

        def __init__(self, image_set, label_set, transform=None, rot=False):
            '''
            Args:
                image_set (numpy int): matrix containing images of 
                rot-MIST digits
                label_set (numpy int): matrix containing labels for
                the digits
            '''
            self.image_set = image_set
            self.label_set = label_set
            self.transform = transform
            self.rot = rot

        def __len__(self):
            '''
            Returns length of image_set
            '''
            return len(self.image_set)

        def __getitem__(self, idx):
            '''
            Behavior: Takes a random index from the instance of Dataloader and returns
            the respective sample from the data
            Args:
                idx (int): denotes index of sample to be returned
            Returns:
                image_sample (torch tensor): 1D matrix containing the image sample
                label_sample (torch tensor): label of the respective sample
            '''

            image_sample = self.image_set[idx].astype(np.float32)
            label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

            if self.transform is not None:
                if self.rot:
                    angle_in_degrees = np.random.rand() * 360.
                    image_sample = ndimage.rotate(image_sample, angle_in_degrees, reshape=False, mode='reflect', order=2)
                image_sample = self.transform(image_sample)

            return image_sample, label_sample

    trans_train = transforms.Compose([transforms.ToTensor()])

    trans_valid = transforms.Compose([transforms.ToTensor()])

    # creating train_loader and valid_loader
    train_dataset = RotMNISTDataset(X_train, y_train, transform=trans_train, rot=False)
    valid_dataset = RotMNISTDataset(X_test, y_test, transform=trans_valid, rot=False)
    print(len(train_dataset), len(valid_dataset))

    batch_size = 64

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=4)

    return trainloader, validloader


def getMNIST(train_size, seed):

    # Loading data
    images_folder_train = 'D:/datasets/MNIST/train.amat'
    images_folder_test = 'D:/datasets/MNIST/test.amat'
    #images_folder_train = 'C:/Users/vdelchev/Documents/datasets/MNIST/train.amat'
    #images_folder_test = 'C:/Users/vdelchev/Documents/datasets/MNIST/test.amat'

    #images_folder_train = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST/train.amat'
    #images_folder_test = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST/test.amat'

    train = np.loadtxt(images_folder_train)
    test = np.loadtxt(images_folder_test)

    all_data = np.concatenate((train, test), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(all_data[:,0:784], all_data[:,-1], 
                                                        test_size=1.-train_size, 
                                                        random_state=seed, stratify=all_data[:,-1])

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    class RotMNISTDataset(Dataset):
        '''Rot-MNIST Dataset'''

        def __init__(self, image_set, label_set, transform=None, rot=False):
            '''
            Args:
                image_set (numpy int): matrix containing images of 
                rot-MIST digits
                label_set (numpy int): matrix containing labels for
                the digits
            '''
            self.image_set = image_set
            self.label_set = label_set
            self.transform = transform
            self.rot = rot

        def __len__(self):
            '''
            Returns length of image_set
            '''
            return len(self.image_set)

        def __getitem__(self, idx):
            '''
            Behavior: Takes a random index from the instance of Dataloader and returns
            the respective sample from the data
            Args:
                idx (int): denotes index of sample to be returned
            Returns:
                image_sample (torch tensor): 1D matrix containing the image sample
                label_sample (torch tensor): label of the respective sample
            '''

            image_sample = self.image_set[idx].astype(np.float32)
            label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

            if self.transform is not None:
                if self.rot:
                    angle_in_degrees = np.random.rand() * 360.
                    image_sample = ndimage.rotate(image_sample, angle_in_degrees, reshape=False, mode='reflect', order=2)
                image_sample = self.transform(image_sample)

            return image_sample, label_sample

    trans_train = transforms.Compose([transforms.ToTensor()])

    trans_valid = transforms.Compose([transforms.ToTensor()])

    # creating train_loader and valid_loader
    train_dataset = RotMNISTDataset(X_train, y_train, transform=trans_train, rot=False)
    valid_dataset = RotMNISTDataset(X_test, y_test, transform=trans_valid, rot=True)
    print(len(train_dataset), len(valid_dataset))

    batch_size = 64

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=0)

    return trainloader, validloader


def getMNISTBACK(train_size, seed):

    # Loading data
    images_folder_train = 'D:/datasets/MNIST_back/train.amat'
    images_folder_test = 'D:/datasets/MNIST_back/test.amat'
    #images_folder_train = 'C:/Users/vdelchev/Documents/datasets/MNIST_back/train.amat'
    #images_folder_test = 'C:/Users/vdelchev/Documents/datasets/MNIST_back/test.amat'

    #images_folder_train = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST_back/train.amat'
    #images_folder_test = '/gpfs/scratch/acad/bcnn/datasets_JMLR/MNIST_back/test.amat'

    train = np.loadtxt(images_folder_train)
    test = np.loadtxt(images_folder_test)

    all_data = np.concatenate((train, test), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(all_data[:,0:784], all_data[:,-1], 
                                                        test_size=1.-train_size, 
                                                        random_state=seed, stratify=all_data[:,-1])

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    class RotMNISTDataset(Dataset):
        '''Rot-MNIST Dataset'''

        def __init__(self, image_set, label_set, transform=None, rot=False):
            '''
            Args:
                image_set (numpy int): matrix containing images of 
                rot-MIST digits
                label_set (numpy int): matrix containing labels for
                the digits
            '''
            self.image_set = image_set
            self.label_set = label_set
            self.transform = transform
            self.rot = rot

        def __len__(self):
            '''
            Returns length of image_set
            '''
            return len(self.image_set)

        def __getitem__(self, idx):
            '''
            Behavior: Takes a random index from the instance of Dataloader and returns
            the respective sample from the data
            Args:
                idx (int): denotes index of sample to be returned
            Returns:
                image_sample (torch tensor): 1D matrix containing the image sample
                label_sample (torch tensor): label of the respective sample
            '''

            image_sample = self.image_set[idx].astype(np.float32)
            label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

            if self.transform is not None:
                if self.rot:
                    angle_in_degrees = np.random.rand() * 360.
                    image_sample = ndimage.rotate(image_sample, angle_in_degrees, reshape=False, mode='reflect', order=2)
                image_sample = self.transform(image_sample)

            return image_sample, label_sample

    trans_train = transforms.Compose([transforms.ToTensor(),
                                      #transforms.Normalize((0.5), (0.5))
                                      ])

    trans_valid = transforms.Compose([transforms.ToTensor(),
                                      #transforms.Normalize((0.5), (0.5))
                                      #transforms.RandomRotation(180, interpolation=InterpolationMode.BILINEAR, expand=True),
                                      #transforms.CenterCrop(28)
                                      ])

    # creating train_loader and valid_loader
    train_dataset = RotMNISTDataset(X_train, y_train, transform=trans_train, rot=False)
    valid_dataset = RotMNISTDataset(X_test, y_test, transform=trans_valid, rot=True)
    print(len(train_dataset), len(valid_dataset))

    batch_size = 64

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=0)

    return trainloader, validloader


def getGalaxy(train_size, seed):

    # Loading data
    #images = 'D:/datasets/galaxy10/Galaxy10_DECals_128.npy'
    #labels = 'D:/datasets/galaxy10/Galaxy10_DECals_128_labels.npy'

    images = '/gpfs/scratch/acad/bcnn/datasets_JMLR/galaxy10/Galaxy10_DECals_128.npy'
    labels = '/gpfs/scratch/acad/bcnn/datasets_JMLR/galaxy10/Galaxy10_DECals_128_labels.npy'

    X = np.load(images)
    y = np.load(labels)

    print(y.shape, y.min(), y.max())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.-train_size, random_state=seed, stratify=y)
    del X, y

    class GalaxyDataset(Dataset):
        '''Galaxy Dataset'''

        def __init__(self, image_set, label_set, transform=None, rot=False):
            self.image_set = image_set
            self.label_set = label_set
            self.transform = transform
            self.rot = rot

        def __len__(self):
            return len(self.image_set)

        def __getitem__(self, idx):
            image_sample = self.image_set[idx].astype(np.float32)
            label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

            if self.transform is not None:
                if self.rot:
                    angle_in_degrees = np.random.rand() * 360.
                    image_sample = ndimage.rotate(image_sample, angle_in_degrees, reshape=False, mode='reflect', order=2)
                image_sample = self.transform(image_sample)

            return image_sample, label_sample

    trans_train = transforms.Compose([transforms.ToTensor(),
                                      #transforms.RandomRotation(360, interpolation=InterpolationMode.BILINEAR),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5)])

    trans_valid = transforms.Compose([transforms.ToTensor()])

    # creating train_loader and valid_loader
    train_dataset = GalaxyDataset(X_train, y_train, transform=trans_train, rot=True)
    valid_dataset = GalaxyDataset(X_test, y_test, transform=trans_valid, rot=False)
    print(len(train_dataset), len(valid_dataset))

    batch_size = 16

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=4)

    return trainloader, validloader


def getGalaxyNOAUG(train_size, seed):

    # Loading data
    #images = 'D:/datasets/galaxy10/Galaxy10_DECals_128.npy'
    #labels = 'D:/datasets/galaxy10/Galaxy10_DECals_128_labels.npy'

    images = '/gpfs/scratch/acad/bcnn/datasets_JMLR/galaxy10/Galaxy10_DECals_128.npy'
    labels = '/gpfs/scratch/acad/bcnn/datasets_JMLR/galaxy10/Galaxy10_DECals_128_labels.npy'

    X = np.load(images)
    y = np.load(labels)

    print(y.shape, y.min(), y.max())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.-train_size, random_state=seed, stratify=y)
    del X, y

    class GalaxyDataset(Dataset):
        '''Galaxy Dataset'''

        def __init__(self, image_set, label_set, transform=None, rot=False):
            self.image_set = image_set
            self.label_set = label_set
            self.transform = transform
            self.rot = rot

        def __len__(self):
            return len(self.image_set)

        def __getitem__(self, idx):
            image_sample = self.image_set[idx].astype(np.float32)
            label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

            if self.transform is not None:
                if self.rot:
                    angle_in_degrees = np.random.rand() * 360.
                    image_sample = ndimage.rotate(image_sample, angle_in_degrees, reshape=False, mode='reflect', order=2)
                image_sample = self.transform(image_sample)

            return image_sample, label_sample

    trans_train = transforms.Compose([transforms.ToTensor()])

    trans_valid = transforms.Compose([transforms.ToTensor()])

    # creating train_loader and valid_loader
    train_dataset = GalaxyDataset(X_train, y_train, transform=trans_train, rot=False)
    valid_dataset = GalaxyDataset(X_test, y_test, transform=trans_valid, rot=False)
    print(len(train_dataset), len(valid_dataset))

    batch_size = 16

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=4)

    return trainloader, validloader


def getMalaria(train_size, seed):

    # Loading
    images = 'C:/Users/vdelchev/Documents/datasets/Malaria/X.npy'
    labels = 'C:/Users/vdelchev/Documents/datasets/Malaria/labels.npy'

    #images = '/gpfs/scratch/acad/bcnn/datasets_JMLR/Malaria/X.npy'
    #labels = '/gpfs/scratch/acad/bcnn/datasets_JMLR/Malaria/labels.npy'

    X = np.load(images)
    y = np.load(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.-train_size, random_state=seed, stratify=y)
    del X, y

    class MalariaDataset(Dataset):
        '''Galaxy Dataset'''

        def __init__(self, image_set, label_set, transform=None, rot=False):
            self.image_set = image_set
            self.label_set = label_set
            self.transform = transform
            self.rot = rot

        def __len__(self):
            return len(self.image_set)

        def __getitem__(self, idx):
            image_sample = self.image_set[idx].astype(np.float32)
            label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

            if self.transform is not None:
                if self.rot:
                    angle_in_degrees = np.random.rand() * 360.
                    image_sample = ndimage.rotate(image_sample, angle_in_degrees, reshape=False, mode='reflect', order=2)
                image_sample = self.transform(image_sample)

            return image_sample, label_sample

    trans_train = transforms.Compose([transforms.ToTensor(),
                                      #transforms.RandomRotation(360, interpolation=InterpolationMode.BILINEAR),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5)])

    trans_valid = transforms.Compose([transforms.ToTensor()])

    # creating train_loader and valid_loader
    train_dataset = MalariaDataset(X_train, y_train, transform=trans_train, rot=True)
    valid_dataset = MalariaDataset(X_test, y_test, transform=trans_valid, rot=False)
    print(len(train_dataset), len(valid_dataset))

    batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=0)

    return trainloader, validloader


def getMalariaNOAUG(train_size, seed):

    # Loading
    #images = 'D:/datasets/Malaria/X.npy'
    #labels = 'D:/datasets/Malaria/labels.npy'

    images = '/gpfs/scratch/acad/bcnn/datasets_JMLR/Malaria/X.npy'
    labels = '/gpfs/scratch/acad/bcnn/datasets_JMLR/Malaria/labels.npy'

    X = np.load(images)
    y = np.load(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.-train_size, random_state=seed, stratify=y)
    del X, y

    class MalariaDataset(Dataset):
        '''Galaxy Dataset'''

        def __init__(self, image_set, label_set, transform=None, rot=False):
            self.image_set = image_set
            self.label_set = label_set
            self.transform = transform
            self.rot = rot

        def __len__(self):
            return len(self.image_set)

        def __getitem__(self, idx):
            image_sample = self.image_set[idx].astype(np.float32)
            label_sample = torch.from_numpy(np.asarray(self.label_set[idx]).astype(np.float32))

            if self.transform is not None:
                if self.rot:
                    angle_in_degrees = np.random.rand() * 360.
                    image_sample = ndimage.rotate(image_sample, angle_in_degrees, reshape=False, mode='reflect', order=2)
                image_sample = self.transform(image_sample)

            return image_sample, label_sample

    trans_train = transforms.Compose([transforms.ToTensor()])

    trans_valid = transforms.Compose([transforms.ToTensor()])

    # creating train_loader and valid_loader
    train_dataset = MalariaDataset(X_train, y_train, transform=trans_train, rot=False)
    valid_dataset = MalariaDataset(X_test, y_test, transform=trans_valid, rot=False)
    print(len(train_dataset), len(valid_dataset))

    batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=4)

    return trainloader, validloader

def getBigEarthNet(train_size, seed):

    path = 'C:/Users/vdelchev/Documents/datasets/bigearthnet/'
    #path = '/gpfs/scratch/acad/bcnn/datasets_JMLR/'

    from bigearthnet_datamodule import BigEarthNetHubDataset

    trans_train = transforms.Compose([transforms.ToTensor(),
                                      #transforms.RandomRotation(360, interpolation=InterpolationMode.BILINEAR),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    trans_valid = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    if train_size < 0.008:
        train_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-debug/bigearthnet-debug/', rot=True, train=True, transforms=trans_train)
        valid_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-debug/bigearthnet-debug/', rot=False, train=False, transforms=trans_valid)
    elif train_size == 0.008:
        train_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-mini/bigearthnet-mini/', rot=True, train=True, transforms=trans_train)
        valid_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-mini/bigearthnet-mini/', rot=False, train=False, transforms=trans_valid)
    elif train_size == 0.08:
        train_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-medium/bigearthnet-medium/', rot=True, train=True, transforms=trans_train)
        valid_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-medium/bigearthnet-medium/', rot=False, train=False, transforms=trans_valid)
    elif train_size == 0.8:
        train_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-full/bigearthnet-full/', rot=True, train=True, transforms=trans_train)
        valid_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-full/bigearthnet-full/', rot=False, train=False, transforms=trans_valid)

    batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=0)

    return trainloader, validloader


def getBigEarthNetNOAUG(train_size, seed):

    path = 'C:/Users/vdelchev/Documents/datasets/bigearthnet/'
    #path = '/gpfs/scratch/acad/bcnn/datasets_JMLR/'

    from bigearthnet_datamodule import BigEarthNetHubDataset

    trans_train = transforms.Compose([transforms.ToTensor(),
                                      #transforms.RandomRotation(360, interpolation=InterpolationMode.BILINEAR),
                                      #transforms.RandomHorizontalFlip(p=0.5),
                                      #transforms.RandomVerticalFlip(p=0.5),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])

    trans_valid = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    if train_size < 0.008:
        train_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-debug/bigearthnet-debug/', rot=False, train=True, transforms=trans_train)
        valid_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-debug/bigearthnet-debug/', rot=False, train=False, transforms=trans_valid)
    elif train_size == 0.008:
        train_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-mini/bigearthnet-mini/', rot=False, train=True, transforms=trans_train)
        valid_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-mini/bigearthnet-mini/', rot=False, train=False, transforms=trans_valid)
    elif train_size == 0.08:
        train_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-medium/bigearthnet-medium/', rot=False, train=True, transforms=trans_train)
        valid_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-medium/bigearthnet-medium/', rot=False, train=False, transforms=trans_valid)
    elif train_size == 0.8:
        train_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-full/bigearthnet-full/', rot=False, train=True, transforms=trans_train)
        valid_dataset = BigEarthNetHubDataset(dataset_path=path+'bigearthnet-full/bigearthnet-full/', rot=False, train=False, transforms=trans_valid)

    batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=4)

    return trainloader, validloader


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))