import numpy as np
import pickle as p
import h5py

class EndOfBatchError(IndexError):
    pass

class BatchFetcher:
    def __init__(self, dataset, labels, down_sample=100):
        self._N = dataset.shape[0]
        _perm = np.random.permutation(self._N)
        self._dataset = dataset[_perm]
        self._labels = labels[_perm]
        self._curri = 0
        self.down_sample = down_sample

    def next_batch(self, batch_size):
        assert self._N > batch_size
        curri = self._curri
        endi = curri + batch_size
        if endi > self._N:
            raise EndOfBatchError
        else:
            inds = np.arange(curri, endi)
            perm = np.random.permutation(self._dataset.shape[1])[::self.down_sample]
        self._curri = endi
        return self._dataset[np.ix_(inds,perm)], self._labels[inds]

class DatasetFetcher:
    def __init__(self, path, batch_size=128):
        with h5py.File(path, 'r') as f:
            _train_data = np.array(f['tr_cloud'])
            _train_label = np.array(f['tr_labels'])
            _test_data = np.array(f['test_cloud'])
            _test_label = np.array(f['test_labels'])
        # self.train = BatchFetcher(_train_data, _train_label)
        # self.validation = BatchFetcher()
        # self.test = BatchFetcher(_test_data, _test_label)
        self.batch_size = batch_size

        self.train_batch = lambda b=self.batch_size: self.batch_data(
            _train_data, _train_label, b
        )
        self.test_batch = lambda b=self.batch_size: self.batch_data(
            _test_data, _test_label, b
        )
        # self.validation_batch = lambda b=self.batch_size: self.validation.next_batch(b)

    def batch_data(self, data, labels, batch_size):
        fetcher = BatchFetcher(data, labels)
        try:
            while(True):
                yield fetcher.next_batch(batch_size)
        except EndOfBatchError:
            pass

