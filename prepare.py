import os
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
    num = 0

    for x,y in zip(file_List ,name_List):
        if num==num_train:
            break
        else:
            img_path =Path+str(x)
            img = PIL.Image.open(img_path).convert("RGB")
            transform = transforms.Compose(
                [transforms.Scale((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.25,0.25,0.25))]
            )
            A = transform(img).to(device)
            train_Data.append((A,y))
            num +=1

    return torch.utils.data.DataLoader(train_Data,batch_size = batch_size,shuffle=shuffle)
