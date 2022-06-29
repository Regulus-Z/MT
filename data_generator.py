import os
import numpy as np
import pandas as pd
import random
import logging
from sklearn.utils.class_weight import compute_class_weight

from utilities import read_audio
import config

class DataGenerator(object):
    def __init__(self, dataset_dir, batch_size, dev_train_csv=None, dev_validate_csv=None,audio_name=None,audio_num=None，seed=1234,SpecAugment=None):
        """
        Inputs:
          batch_size: int
          dev_train_csv: str | None, if None then use all data for training
          dev_validate_csv: str | None, if None then use all data for training
          seed: int, random seed
        """
        self.len_dict={'breathing-deep':240000,'breathing-shallow':160000,'cough-heavy':80000,
                       'cough-shallow':72000,'counting-fast':112000,'counting-normal':208000,
                       'vowel-a':160000,'vowel-e':160000,'vowel-o':160000}
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        #self.cycle_len = config.cycle_len
        #####add cycle length of cough
        #self.cycle_len_ch = config.cycle_len_cough_heavy
        #self.cycle_len_va = config.cycle_len_va
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        lb_to_ix = config.lb_to_ix

        # read data
        self.train_audio_names, train_cycle_labels = self.get_audio_info_from_csv(dev_train_csv)
        self.validate_audio_names, validate_cycle_labels = self.get_audio_info_from_csv(dev_validate_csv)
        
        #read positive data for SpecAugment
        #self.pos_sample_names,pos_sample_labels=self.get_audio_info_from_csv(SpecAugment)
            
        self.train_y = np.array([lb_to_ix[train_cycle_labels[lb]] for lb in range(0, len(train_cycle_labels))])
        self.validate_y = np.array([lb_to_ix[validate_cycle_labels[lb]] for lb in range(0, len(validate_cycle_labels))])
                
        logging.info('The data contains: {} training and {} validation data.'.format(len(self.train_audio_names), len(self.validate_audio_names)))

    def calculate_class_weight(self):
        classes_ix = np.sort(np.unique(self.train_y))
        class_weight = compute_class_weight(class_weight='balanced', classes=classes_ix, y=self.train_y)
        #return class_weight
        
        return np.power(class_weight, 2)

    def get_audio_info_from_csv(self, csv_file):
        """Read a csv file.
        Args:
          csv_file: str, path of csv file
        """
        
        df = pd.read_csv(csv_file, sep=',')
        df = pd.DataFrame(df)

        audio_names = []
        cycle_labels = []

        for row in df.iterrows():
            audio_name = row[1]['id']
            cycle_label = row[1]['covid_status']

            audio_names.append(audio_name)
            cycle_labels.append(cycle_label)

        return audio_names, cycle_labels

    def read_batch_audio(self, batch_audio_indexes, data_type, generate_type):

        
        batch_audio = []
        for i in range(audio_num):
            batch_audio.append([])

        batch_audio_name = []
        if data_type == 'train':
            audio_names_list = self.train_audio_names
        elif data_type == 'evaluate':
            audio_names_list = self.validate_audio_names

        
        for ind in range(0, len(batch_audio_indexes)):
            audio_name = audio_names_list[batch_audio_indexes[ind]]
            batch_audio_name.append(audio_name)
            #########################
            for i in range(audio_num):
                audio, fs = read_audio(os.path.join(self.dataset_dir,audio_name[i], 'audio', audio_name))
                cycle_len = self.len_dict[audio_name[i]]
            #########################
                if len(audio) > cycle_len:
                    if generate_type == 'ge_train':
                        start_ind = random.randint(0, len(audio) - cycle_len)
                    elif generate_type == 'ge_validate':
                        start_ind = int((len(audio) - cycle_len)/2)
                    else:
                        print('Wrong data generation type')
                    audio_pad = audio[start_ind:start_ind + cycle_len]
                elif len(audio) < cycle_len:
                    audio_pad = np.pad(audio, (0, cycle_len-len(audio)), mode='wrap')
                elif len(audio) == cycle_len:
                    audio_pad = audio
                else:
                    print('Wrong audio length!')
                batch_audio[i].append(audio_pad)
    
        return batch_audio,batch_audio_name

    def generate_train(self):
        """Generate mini-batch data for training.
        Returns:
          batch_x: (batch_size, seq_len, freq_bins,audio_num)
          batch_y: (batch_size,)
        """

        batch_size = self.batch_size
        audio_indexes = np.arange(len(self.train_audio_names))
        audios_num = len(audio_indexes)
        
        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0
        
        while True:
            # Reset pointer
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)
            
            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size
            
            iteration += 1
        
            batch_x,batch_audio_names = self.read_batch_audio(batch_audio_indexes, 'train', 'ge_train')
            batch_x = np.array(batch_x)
            batch_y = self.train_y[batch_audio_indexes]

            yield batch_x,batch_y, batch_audio_names

    def generate_validate(self, data_type, shuffle, max_iteration=None):
        """Generate mini-batch data for evaluation. 
        
        Args:
          data_type: 'train' | 'evaluate'
          max_iteration: int, maximum iteration for validation
          shuffle: bool
          
        Returns:
          batch_x: (batch_size, seq_len, freq_bins,audio_num)
          batch_audio_names: (batch_size,)
        """
        # Due to different sizes of the data at the evaluation stage, we set batch_size = 1
        batch_size = self.batch_size

        if data_type == 'train':
            audio_indexes = np.arange(len(self.train_audio_names))
        elif data_type == 'evaluate':
            audio_indexes = np.arange(len(self.validate_audio_names))
        else:
            raise Exception('Invalid data_type!')
            
        if shuffle:
            self.validate_random_state.shuffle(audio_indexes)

        audios_num = len(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
                
            pointer += batch_size
            
            iteration += 1

            batch_x,batch_audio_names = self.read_batch_audio(batch_audio_indexes, data_type, 'ge_validate')
            batch_x = np.array(batch_x)
            if data_type == 'train':
                batch_y = self.train_y[batch_audio_indexes]
            elif data_type == 'evaluate':
                batch_y = self.validate_y[batch_audio_indexes]
            else:
                raise Exception('Invalid data_type!')

            yield batch_x,batch_y, batch_audio_names

        
