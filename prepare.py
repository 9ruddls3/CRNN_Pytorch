import os
import argparse
import shutil
import sys
import PIL
from tqdm import tqdm

import torch
from torchvision import transforms

from parms import batch_size as b_size
from parms import shuffle as sfle

def Preparing(In, Out, file_name, batch_size, shuffle=False):
    AbsDir = os.path.abspath(In)
    file_List = []

    for x in os.listdir(AbsDir):
        file_List.append(x)

    file_List.sort()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_Data = []
    file_name = file_name +'.pth'
    
    if len(file_List)//batch_size !=0:
        num_dataset= (len(file_List)//batch_size)*batch_size
        pass
    else:
        num_dataset= len(file_List)
        pass
    file_list = file_list[:num_dataset]
    
    for x in tqdm(list(enumerate(file_List))):
        img_path = In + '/' + str(x[1])
        img = PIL.Image.open(img_path).convert("RGB")
        transform = transforms.Compose(
            [transforms.Scale((224, 224)), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))]
        )
        A = transform(img).to(device)
        train_Data.append((A, x[1][:-4]))

    
    return torch.save(torch.utils.data.DataLoader(train_Data, batch_size=batch_size, shuffle=shuffle), Out + '/' + file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--In_dir', type=str, required=True, help='path to folder which contains the images')
    parser.add_argument('--Out_dir', type=str, required=True, help='Torch_Dataloader output path')

    args = parser.parse_args()
    from_path = os.path.abspath(args.In_dir)
    to_path = os.path.abspath(args.Out_dir)

    while True:
        print("Enter Dataloader file name (.pth)")
        f_name = str(input())
        dir = from_path +'/' + f_name
        if os.path.isfile(dir):
            print("Please Enter other name (it existed already)", end='\n')
        else:
            Preparing(from_path, to_path, f_name, b_size, sfle)
            print('DataLoader Creation Complete!')
            sys.exit()

