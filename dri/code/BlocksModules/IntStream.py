from fuel.streams import AbstractDataStream
from fuel.datasets import IterableDataset, IndexableDataset
from fuel.iterator import DataIterator
from fuel.schemes import ShuffledScheme


class IntStream(AbstractDataStream):
    """A stream of data from integers all the way up to maxBatch.
    Parameters
    ----------
    maxBatch : maximum integer to be reached
    """
    def __init__(self, startIndex, numExamples, batchSize, name, **kwargs):
        super(IntStream, self).__init__(**kwargs)
        self.startIndex=startIndex
        self.count=startIndex-batchSize
        self.numExamples = numExamples+startIndex
        self.batchSize = batchSize
        self._sources=[name+"_From", name+"_To"]

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.reset()

    def reset(self):
        self.count=self.startIndex-1

    def next_epoch(self):
        self.count+=self.batchSize
        if self.count>=self.numExamples:
            self.reset()
            raise StopIteration

    def get_data(self, request=None):
        """Get data from the dataset."""
        ret = self.count
        try:
            self.next_epoch()
        except StopIteration:
            raise StopIteration
        return [self.count, min(self.count+self.batchSize, self.numExamples)]

    def get_epoch_iterator(self, **kwargs):
        """Get an epoch iterator for the data stream."""
        return super(IntStream, self).get_epoch_iterator(**kwargs)

if __name__ == "__main__":
    startIndex = 0
    numExamples = 50
    batchSize = 1
    stream = IntStream(startIndex, numExamples, batchSize, 'train')
    print(stream.sources)
    siter = stream.get_epoch_iterator()
    counter=0
    while True:
        counter+=1
        if counter==2000:
            break
        try:
            batch = next(siter)
            print(batch)
        except StopIteration:
            print("stop")
            
    print('here')