import os
import os.path as osp
import sys
import pickle
from PIL import Image
from glob import glob
import numpy as np
import torch
import torch.utils.data.distributed
from torchvision import transforms
from torchvision.datasets.utils import download_url, check_integrity


class DebugDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(DebugDataset, self).__init__()

        self.images_id = np.array([0, 1]).astype(np.int64)
        self.images = np.array([[0.1, 0.1], [0.9, 0.9]]).astype(np.float32)
        self.labels = np.array([0, 1]).astype(np.int64)

    def __getitem__(self, index):
        return self.images_id[index], self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images_id)


class ImageNetIDDataset(torch.utils.data.Dataset):
    def __init__(self, root_dirname, phase, seed, size=224):
        super(ImageNetIDDataset, self).__init__()
        assert phase in ['train', 'val', 'valv2']

        # get 1000 classes
        image_dirname = osp.join(root_dirname, phase)  # e.g., data/imagenet/val
        classes = [d for d in os.listdir(image_dirname) if os.path.isdir(os.path.join(image_dirname, d))]
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        assert len(self.class_to_idx) == 1000

        # get all images
        self.images_fname = glob('{}/*/*.JPEG'.format(image_dirname))
        self.images_fname.sort()
        if phase == 'train':
            assert len(self.images_fname) == 1281167
        elif phase == 'val':
            assert len(self.images_fname) == 50000
        elif phase == 'valv2':
            assert len(self.images_fname) == 80000
            # move imagenetv2 images in the front, and imagenet val images into the back
            imagenetv2_images_fname = list(filter(lambda t: t.split('/')[-1].startswith('imagenetv2-'),
                                                  self.images_fname))
            imagenet_val_images_fname = list(filter(lambda t: not t.split('/')[-1].startswith('imagenetv2-'),
                                                    self.images_fname))
            assert len(imagenetv2_images_fname) == 30000
            assert len(imagenet_val_images_fname) == 50000
            self.images_fname = imagenetv2_images_fname + imagenet_val_images_fname
            assert len(self.images_fname) == 80000
        else:
            raise NotImplementedError('Unknown phase {} for imagenet support phases are: train/val/valv2'.format(phase))

        # record the index of each image in the whole dataset, since we may select a subset later
        self.images_id = np.arange(len(self.images_fname))

        # fix random seed
        # save previous RNG state
        state = np.random.get_state()
        # fix random seed, thus we have the same random status at each time
        np.random.seed(seed)
        if phase == 'valv2':
            perm1 = np.random.permutation(30000)  # for imagenetv2 images
            perm2 = np.random.permutation(50000) + 30000  # for imagenet val images
            perm = np.hstack((perm1, perm2))  # imagenetv2 goes before imagenet val
            assert len(perm) == len(self.images_fname)
        else:
            perm = np.random.permutation(len(self.images_fname))
        self.images_fname = list(np.array(self.images_fname)[perm])
        self.images_id = list(np.array(self.images_id)[perm])
        # restore previous RNG state for training && test
        np.random.set_state(state)

        # transform
        self.transform = transforms.Compose([transforms.Resize(int(size / 0.875)),  # i.e., 224 / 0.875 = 256
                                             transforms.CenterCrop(size),
                                             transforms.ToTensor()])

    def __getitem__(self, index):
        # we always emit data in [0, 1] range to keep things simpler (normalization is performed in the main script).
        # image_id is the identifier of current image in the whole dataset
        # index is the index of current image in self.images_fname
        image_id = self.images_id[index]
        image_fname = self.images_fname[index]
        label = self.class_to_idx[image_fname.split('/')[-2]]
        with open(image_fname, 'rb') as f:
            with Image.open(f) as image:
                image = image.convert('RGB')

        # get standard format: 224x224, 0-1 range, RGB channel order
        image = self.transform(image)  # [3, 224, 224]

        return image_id, image, label

    def __len__(self):
        return len(self.images_fname)


class CIFAR10IDDataset(torch.utils.data.Dataset):
    # Adopted from torchvision
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root_dirname, phase, seed):
        self.root_dirname = os.path.expanduser(root_dirname)
        self.phase = phase  # training set or test set
        assert self.phase in ['train', 'test']

        # we load CIFAR-10 dataset into standard format (without any data augmentation)
        self.transform = transforms.ToTensor()

        # download if not exists, and check integrity
        self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.phase == 'train':
            self.images = []
            self.labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root_dirname, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.images.append(entry['data'])
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']
                fo.close()

            self.images = np.concatenate(self.images)
            self.images = self.images.reshape((50000, 3, 32, 32))
            self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root_dirname, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.images = entry['data']
            if 'labels' in entry:
                self.labels = entry['labels']
            else:
                self.labels = entry['fine_labels']
            fo.close()
            self.images = self.images.reshape((10000, 3, 32, 32))
            self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC

        # record the index of each image in the whole dataset, since we may select a subset later
        self.images_id = np.arange(len(self.images))

        # fix random seed
        # save previous RNG state
        state = np.random.get_state()
        # fix random seed, thus we have the same random status at each time
        np.random.seed(seed)
        perm = np.random.permutation(len(self.images))
        self.images_id = list(np.array(self.images_id)[perm])
        self.images = list(np.array(self.images)[perm])
        self.labels = list(np.array(self.labels)[perm])
        # restore previous RNG state for training && test
        np.random.set_state(state)

    def __getitem__(self, index):
        image_id, image, label = self.images_id[index], self.images[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        image = self.transform(image)

        return image_id, image, label

    def __len__(self):
        return len(self.images)

    def _check_integrity(self):
        root = self.root_dirname
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root_dirname = self.root_dirname
        download_url(self.url, root_dirname, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root_dirname, self.filename), "r:gz")
        os.chdir(root_dirname)
        tar.extractall()
        tar.close()
        os.chdir(cwd)


class MNISTIDDataset(torch.utils.data.Dataset):
    # Adopted from torchvision
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root_dirname, phase, seed):
        self.root_dirname = os.path.expanduser(root_dirname)
        self.phase = phase  # training set or test set
        assert self.phase in ['train', 'test']

        # we load MNIST dataset into standard format (without any data augmentation)
        self.transform = transforms.ToTensor()

        # download if not exists, and check integrity
        self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.phase == 'train':
            self.images, self.labels = torch.load(
                os.path.join(self.root_dirname, self.processed_folder, self.training_file))
        else:
            self.images, self.labels = torch.load(
                os.path.join(self.root_dirname, self.processed_folder, self.test_file))

        # record the index of each image in the whole dataset, since we may select a subset later
        self.images_id = np.arange(len(self.images))

        # fix random seed
        # save previous RNG state
        state = np.random.get_state()
        # fix random seed, thus we have the same random status at each time
        np.random.seed(seed)
        perm = np.random.permutation(len(self.images))
        self.images_id = list(np.array(self.images_id)[perm])
        self.images = list(np.array(self.images)[perm])
        self.labels = list(np.array(self.labels)[perm])
        # restore previous RNG state for training && test
        np.random.set_state(state)

    def __getitem__(self, index):
        image_id, image, label = self.images_id[index], self.images[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image, mode='L')

        image = self.transform(image)

        return image_id, image, label

    def __len__(self):
        return len(self.images)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root_dirname, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root_dirname, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root_dirname, self.raw_folder))
            os.makedirs(os.path.join(self.root_dirname, self.processed_folder))
        except OSError as e:
            import errno
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root_dirname, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        def get_int(b):
            import codecs
            return int(codecs.encode(b, 'hex'), 16)

        def read_label_file(path):
            with open(path, 'rb') as f:
                data = f.read()
                assert get_int(data[:4]) == 2049
                length = get_int(data[4:8])
                parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
                return torch.from_numpy(parsed).view(length).long()

        def read_image_file(path):
            with open(path, 'rb') as f:
                data = f.read()
                assert get_int(data[:4]) == 2051
                length = get_int(data[4:8])
                num_rows = get_int(data[8:12])
                num_cols = get_int(data[12:16])
                images = []
                parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
                return torch.from_numpy(parsed).view(length, num_rows, num_cols)

        training_set = (
            read_image_file(os.path.join(self.root_dirname, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root_dirname, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root_dirname, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root_dirname, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root_dirname, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root_dirname, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


def make_loader(dataset, phase, batch_size, seed, num_worker=0, **kwargs):
    """
    Make loader. To make sure we use the same data for evaluation,
    these loaders will return (image_id, image, label) tuple instead of vanilla (image, label) tuple.
    :param dataset: mnist, cifar10 or imagenet.
    :param phase: train, val or test.
    :param batch_size: batch size. For imagenet we usually set batch size to 1.
    :param seed: random seed in selecting images.
    :param num_worker: number of workers
    :param kwargs: for imagenet, kwargs could contain size (e.g., 224 or 299)
    :return: pytorch DataLoader object, could be used as iterator.
    """

    if dataset == 'imagenet':
        assert phase in ['train', 'val', 'valv2']
        dataset = ImageNetIDDataset('data/imagenet', phase, seed, **kwargs)
        dataset.num_class = 1000
    elif dataset == 'cifar10':
        assert phase in ['train', 'test']
        dataset = CIFAR10IDDataset('data/cifar10', phase, seed)
        dataset.num_class = 10
    elif dataset == 'mnist':
        assert phase in ['train', 'test']
        dataset = MNISTIDDataset('data/mnist', phase, seed)
        dataset.num_class = 10
    elif dataset == 'mnist01':
        assert phase in ['train', 'test']
        dataset = MNISTIDDataset('data/mnist01', phase, seed)
        dataset.num_class = 2
    elif dataset == 'debug':
        dataset = DebugDataset()
        dataset.num_class = 2
    else:
        raise NotImplementedError('Unknown dataset {}'.format(dataset))

    # create loader from dataset
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True)
    loader.num_class = dataset.num_class
    return loader
