import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tensorflow as tf
# A custom dataset handler to iterate the data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data# Perform modification using TensorFlow operations
        last_n_data = self.data['input_sequence'][:, :, -9:]
        modified_data = (last_n_data + 1) / 2

        # Create a new tensor with the modified values
        self.data['input_sequence'] = tf.concat([self.data['input_sequence'][:, :, :-9], modified_data], axis=-1)

    def __getitem__(self, idx):
        
        input_seq = np.array(self.data['input_sequence'][idx])
        output_seq = np.array(self.data['output_sequence'][idx])
        seq_length = np.array(self.data['sequence_length'][idx])
        #control_seq = self.data['control_sequence'][idx]
        if self.data['control_sequence'] is None:
            control_seq = []  # Convert None to an empty list
        return {
            'input_sequence': torch.from_numpy(input_seq).float(),
            'output_sequence': torch.from_numpy(output_seq).float(),
            'sequence_length': torch.from_numpy(seq_length),
            'control_sequence': torch.tensor(control_seq).float()
        }

    def __len__(self):
        return self.data['input_sequence'].shape[0]