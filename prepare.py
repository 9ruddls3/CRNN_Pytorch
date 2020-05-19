import os
import argparse
import shutil
import sys

import torch
from torchvision import transforms

import parms.batch_size as b_size
import parms.num_data as n_data
import parms.shuffle as sfle


def dir_(Dir):
    if Dir.endswith('/')
        return Dir
    else:
        return Dir+'/'
    
def Preparing(from,to,file_name,num_train,batch_size,shuffle=False):
    AbsDir = os.path.abspath(from)
    name_List = []
    file_List = []
    
    for x in os.listdir(AbsDir):
        file_List.append(x)
        name_List.append(x[:-4])

    name_List.sort()
    file_List.sort()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_Data = []
    file_name = file_name+.'pth'

    for x, y in enumerate(list(zip(file_List, name_List))):
        img_path =Path+str(y[0])
        img = PIL.Image.open(img_path).convert("RGB")
        transform = transforms.Compose(
        [transforms.Scale((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.25,0.25,0.25))]
        )
        A = transform(img).to(device)
        train_Data.append((A,y[1]))
        if x==num_train-1:
            break

    return torch.save(torch.utils.data.DataLoader(train_Data,batch_size = batch_size,shuffle=shuffle),to+file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type = str, required = True ,help = 'path to folder which contains the images')
    parser.add_argument('--out_dir', type = str, required = True, help = 'Torch_Dataloader output path')
    
    
    args = parser.parse_args()
    from_path = os.path.abspath(dir_(args.in_dir))
    to_path = os.path.abspath(dir_(args.out_dir))
    
    while True:
        print("Enter Dataloader file name")
        f_name = str(input()) 
        dir = from_path+'/'+f_name
        if os.path.isfile(dir):
            print("Please Enter other name (it existed already)",end=\n)
        else:
            Preparing(dir,f_name,n_data,b_size,sfle)
            print('DataLoader Creation Complete!')
            sys.exit()
    



