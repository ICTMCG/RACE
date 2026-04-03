import numpy as np
from torch.utils.data import Sampler
import logging


class StratifiedBatchSampler(Sampler):
    """
    Stratified Batch Sampler to ensure each batch contains samples from all classes.
    Useful for addressing class imbalance issues.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__(dataset)
        self.dataset = dataset
        # GENERALIZED LABEL ACQUISITION: Works for dict-style datasets
        try:
            self.labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
        except (AttributeError, TypeError):
            raise TypeError("Dataset items must be dictionaries with a 'label' key holding a tensor.")

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_classes = len(np.unique(self.labels))
        self.indices_by_class = [
            np.where(self.labels == i)[0] for i in range(self.num_classes)
        ]

        # Check if any class is empty
        for i in range(self.num_classes):
            if len(self.indices_by_class[i]) == 0:
                # We don't raise an error, but log a warning, as some splits might be missing classes
                logging.getLogger(__name__).warning(f"StratifiedSampler: Class {i} has no samples in the dataset.")

        self.class_counts = [len(i) for i in self.indices_by_class]
        # Filter out empty classes from sampling
        self.active_classes = [i for i, count in enumerate(self.class_counts) if count > 0]
        if not self.active_classes:
            raise ValueError("No samples found for any class.")
        
        self.num_active_classes = len(self.active_classes)

        self.num_batches = len(self.dataset) // self.batch_size
        
        # Calculate samples per class, ensuring at least 1 if possible
        self.samples_per_class = max(1, self.batch_size // self.num_active_classes)

        if self.batch_size < self.num_active_classes:
            logging.getLogger(__name__).warning(
                f"StratifiedSampler: batch_size {self.batch_size} is smaller than the number of active classes {self.num_active_classes}. "
                f"Each batch may not contain all classes."
            )

    def __iter__(self):
        # Pointers to track the current position in each class's index list
        class_pointers = {i: 0 for i in self.active_classes}

        if self.shuffle:
            for i in self.active_classes:
                np.random.shuffle(self.indices_by_class[i])

        for _ in range(self.num_batches):
            batch_indices = []

            # Evenly sample from each active class
            for i in self.active_classes:
                indices = self.indices_by_class[i]
                count = self.class_counts[i]
                pointer = class_pointers[i]

                for _ in range(self.samples_per_class):
                    if pointer >= count: # Reset pointer if we've cycled through all samples
                        if self.shuffle:
                            np.random.shuffle(indices)
                        pointer = 0
                    
                    batch_indices.append(indices[pointer])
                    pointer += 1
                class_pointers[i] = pointer
            
            # Fill remaining spots if batch_size is not perfectly divisible
            remaining_spots = self.batch_size - len(batch_indices)
            if remaining_spots > 0:
                # Draw remaining samples randomly from any active class
                for _ in range(remaining_spots):
                    class_idx = np.random.choice(self.active_classes)
                    indices = self.indices_by_class[class_idx]
                    count = self.class_counts[class_idx]
                    pointer = class_pointers[class_idx]

                    if pointer >= count:
                        if self.shuffle: np.random.shuffle(indices)
                        pointer = 0
                    
                    batch_indices.append(indices[pointer])
                    class_pointers[class_idx] = pointer + 1

            if self.shuffle:
                np.random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches


class AtLeastTwoClassesBatchSampler(Sampler):
    """
    Batch Sampler to ensure each batch contains samples from at least two different classes.
    Useful for contrastive loss functions.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__(dataset)
        self.dataset = dataset
        # GENERALIZED LABEL ACQUISITION
        try:
            self.labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
        except (AttributeError, TypeError):
            raise TypeError("Dataset items must be dictionaries with a 'label' key holding a tensor.")
            
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices_by_class = [
            np.where(self.labels == i)[0] for i in np.unique(self.labels)
        ]
        
        # remove empty classes
        self.indices_by_class = [x for x in self.indices_by_class if len(x) > 0]
        self.num_classes = len(self.indices_by_class)

        if self.num_classes < 2:
            raise ValueError(
                "The dataset must contain at least two classes for this sampler."
            )

        self.class_counts = [len(i) for i in self.indices_by_class]
        self.num_batches = len(self.dataset) // self.batch_size

    def __iter__(self):
        class_pointers = [0] * self.num_classes

        if self.shuffle:
            for i in range(self.num_classes):
                np.random.shuffle(self.indices_by_class[i])

        for _ in range(self.num_batches):
            batch_indices = []

            # 1. Select two different random classes
            class_idx1, class_idx2 = np.random.choice(
                range(self.num_classes), 2, replace=False
            )

            # 2. Add one sample from each of these two classes
            for class_idx in [class_idx1, class_idx2]:
                indices = self.indices_by_class[class_idx]
                pointer = class_pointers[class_idx]

                if pointer >= len(indices):
                    if self.shuffle:
                        np.random.shuffle(indices)
                    pointer = 0
                
                batch_indices.append(indices[pointer])
                class_pointers[class_idx] = pointer + 1

            # 3. Fill the rest of the batch
            remaining_spots = self.batch_size - 2
            for _ in range(remaining_spots):
                # Pick a random class
                class_idx = np.random.randint(self.num_classes)
                indices = self.indices_by_class[class_idx]
                pointer = class_pointers[class_idx]

                if pointer >= len(indices):
                    if self.shuffle:
                        np.random.shuffle(indices)
                    pointer = 0
                
                batch_indices.append(indices[pointer])
                class_pointers[class_idx] = pointer + 1

            if self.shuffle:
                np.random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches
