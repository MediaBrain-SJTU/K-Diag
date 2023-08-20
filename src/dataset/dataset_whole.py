from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from dataset.randaugment import RandomAugment
import pydicom
from skimage import exposure
class Merge_datasets(Dataset):
    def __init__(self, csv_dir , dataset, mode = 'train'):
        if dataset == 'whole':
            data_sets_list = os.listdir(csv_dir)
            try:
                data_info = pd.read_csv(csv_dir + data_sets_list[0])
            except:
                print("Lenth of csv_path_list equals zero!")
            for csv_name in data_sets_list:
                temp_data_info = pd.read_csv( csv_dir + csv_name)
                data_info = pd.merge(data_info,temp_data_info,how='outer')
            data_info.fillna(2,inplace = True)
        else:
            if '.mix' in dataset:
                dataset = dataset.split('.')[0]
                data_info = pd.read_csv( csv_dir + dataset +'.csv') 
                original_pathologies = data_info.columns.values[1:]
                
                data_sets_list = os.listdir(csv_dir)
                for csv_name in data_sets_list:
                    if csv_name ==  (dataset +'.csv'):
                        continue                        
                    temp_data_info = pd.read_csv( csv_dir + csv_name)
                    #temp_pathologies = temp_data_info.columns.values[1:]
                    for pathology in original_pathologies:
                        try:
                            del temp_data_info[pathology]
                        except:
                            continue
                    data_info = pd.merge(data_info,temp_data_info,how='outer')
                data_info.fillna(2,inplace = True)
            else:
                data_info = pd.read_csv( csv_dir + dataset +'.csv')
            
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.pathologies = data_info.columns.values[1:]
        self.class_list = np.asarray(data_info.iloc[:,1:])
        if dataset == 'Physician_label193_all':
            self.pathologies = data_info.columns.values[3:]
            self.class_list = np.asarray(data_info.iloc[:,3:])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if mode == 'train':
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        if mode == 'test':
            self.transform = transforms.Compose([                        
            transforms.Resize([224, 224],interpolation=Image.BICUBIC),  
            transforms.ToTensor(),
            normalize,
            ])   
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index].replace('/mnt/petrelfs/zhangxiaoman/DATA/Chestxray','/mnt/petrelfs/share_data/zhangxiaoman/DATA/ChestXray')
        class_label = self.class_list[index] # (14,)
        # if ('.dicom' in img_path) or ('.dcm' in img_path):
        #     if '.dicom' in img_path:
        #         img = self.read_dicom(img_path) 
        #     else:
        #         img = self.read_dcm(img_path) 
        # else:
        
        ''' for padchest'''
        img_array = np.array(Image.open(img_path))
        img_array = (img_array/img_array.max())*255
        img = Image.fromarray(img_array.astype('uint8')).convert('RGB')
        
        #img = PIL.Image.open(img_path).convert('RGB')   
        image = self.transform(img)

        return {
            "image": image,
            "label": class_label
            }
    # def read_dicom(self, dicom_path):
    #     from pydicom.pixel_data_handlers.util import apply_modality_lut
        
    #     dicom_obj = pydicom.filereader.dcmread(dicom_path)
    #     img = apply_modality_lut(dicom_obj.pixel_array, dicom_obj)
    #     img = img.astype(float) / 255.
    #     img = exposure.equalize_hist(img)
        
    #     img = (255 * img).astype(np.uint8)
    #     img = PIL.Image.fromarray(img).convert('RGB')   
    #     return img
        
    # def read_dcm(self,dcm_path):
    #     dcm_data = pydicom.read_file(dcm_path)
    #     img = dcm_data.pixel_array.astype(float) / 255.
    #     img = exposure.equalize_hist(img)
        
    #     img = (255 * img).astype(np.uint8)
    #     img = PIL.Image.fromarray(img).convert('RGB')   
    #     return img
    
    def __len__(self):
        return len(self.img_path_list)
    
# csv_path_list = '/remote-home/chaoyiwu/classification_to_text/Whole_dataset/train/'

# Dataset = Merge_datasets(csv_path_list)