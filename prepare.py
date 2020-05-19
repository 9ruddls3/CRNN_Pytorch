import os
import argparse
import shutil
import sys

import torch
from torchvision import transforms

def Preparing(Path,num_train,shuffle=False):
    workDIr = os.path.abspath(Path)
    name_List = []
    file_List = []
    for dirpath, dirnames, filenames in os.walk(workDIr):
        for filename in filenames:
            file_List.append(filename)
            name_List.append(filename[:-4])

    name_List.sort()
    file_List.sort()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_Data = []

    for x, y in enumerate(list(zip(file_List, name_List))):
        img_path =Path+str(y[0])
        img = PIL.Image.open(img_path).convert("RGB")
        transform = transforms.Compose(
        [transforms.Scale((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.25,0.25,0.25))]
        )
        A = transform(img).to(device)
        train_Data.append((A,y[1]))
        if x==1999:
            break

    return torch.utils.data.DataLoader(train_Data,batch_size = batch_size,shuffle=shuffle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#     parser.add_argument('--out', type = str, required = True, help = 'Torch_Dataloader output path')
    parser.add_argument('--folder', type = str, required = True ,help = 'path to folder which contains the images')
#     parser.add_argument('--file', type = str, help = 'path to file which contains the image path and label')
    args = parser.parse_args()
    
    workDIr = os.path.abspath(Path)
    name_List = []
    file_List = []
    for dirpath, dirnames, filenames in os.walk(workDIr):
        for filename in filenames:
            print(filename)
    sys.exit()
  
    
    if args.file is not None:
        image_path_list, label_list = read_data_from_file(args.file)
        createDataset(args.out, image_path_list, label_list)
        show_demo(2, image_path_list, label_list)
    elif args.folder is not None:
        image_path_list, label_list = read_data_from_folder(args.folder)
        createDataset(args.out, image_path_list, label_list)
        show_demo(2, image_path_list, label_list)
    else:
        print ('Please use --floder or --file to assign the input. Use -h to see more.')
        sys.exit()



