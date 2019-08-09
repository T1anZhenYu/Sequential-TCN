import torch
from torch.utils.data.dataset import Dataset
import os 
import numpy as np 
class SmartWall(Dataset):
    def __init__(self, data_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        #修改当前工作目录
        os.chdir(data_path)
        #将该文件夹下的所有文件名存入一个列表
        file_list = os.listdir()

        self.data = np.array([[[]]])
        self.label = np.array([])
        for i in range(len(file_list)):
            data_raw = dict(np.load(data_path+'/'+file_list[i],allow_pickle=True))['arr_0']
            for j in range(len(data_raw)):   
                print(self.data.shape)
                print(np.expand_dims(data_raw[j]['data'],axis=0).shape)
                self.data=np.concatenate((self.data,np.expand_dims(data_raw[j]['data'],axis=0)),axis=0)
                self.label=np.concatenate((self.label,data_raw[j]['label']),axis=0) 
        print('type data dim 1 ',type(self.data[0]))
        print('type data dim 3',type(self.data[0][0][0]))
        self.data = self.data
        print('type data',type(self.data))
        self.data_len=len(self.data)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_data = self.data[index]
        # Open image
        singlelabel = self.label[index]

        return (single_data, singlelabel)

    def __len__(self):
        return self.data_len

def data_loader(batch_size):
    train_root='/content/train'
    test_root='/content/test'
    train_dataset=SmartWall(train_root)
    test_dataset=SmartWall(test_root)
    train_loader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    test_loader=torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader,test_loader