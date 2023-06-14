
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniL2Method
from torch.optim import Adam
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
import numpy as np, cv2
import argparse
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension
from dotenv import load_dotenv
load_dotenv()
project_root = os.getenv("PROJECT_ROOT")


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

class CustomImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.extensions = IMG_EXTENSIONS
        self.samples = self.make_dataset(self.root, {'.': 0}, self.extensions)
        self.targets = [0] * len(self.samples)

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        instances = []
        _, _, filenames = next(os.walk(directory))
        
        for filename in sorted(filenames):
            if extensions is not None:
                if not has_file_allowed_extension(filename, extensions):
                    continue
            elif is_valid_file is not None:
                if not is_valid_file(filename):
                    continue

            path = os.path.join(directory, filename)
            item = path, class_to_idx['.']
            instances.append(item)
        return instances


import random
def truncate_dataset(dataset, new_length, original_length):
    ## We need to set the seed or else the results would vary after each run.
    random.seed(42)
    random_index_list = random.sample(range(0,original_length),new_length, )

    return torch.utils.data.Subset(dataset, random_index_list)


model_load_root =  f"{project_root}/weights/classifiers/final_iteration.pt"



## Needs to be downscaled to support resnets dense layers, which require 224,224 image sizes
resnet_im_size = (224,224)
dataset_batch_size=32


img_transforms = transforms.Compose([
    transforms.Resize(resnet_im_size),
    transforms.ToTensor(),
    ])

def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

int_labels = [
            "inanimate",
            "pedestrian",
            "vehicle",
        ]
def get_trained_resnet_model(layers,out_features, weight_str=None, project_root=None):
    

    if layers == 18:
        model_str = "resnet-18"
        torch_in_features = 512
        resnet_model = models.resnet18(weights=None)
    elif layers == 101:
        model_str = "resnet-101"
        torch_in_features = 2048
        resnet_model = models.resnet101(weights=None)
    else:
        raise Exception(f"No trained models found for value {layers}")
        
    resnet_model.fc = torch.nn.Linear(in_features=torch_in_features, out_features=out_features)
    if weight_str != None:
        resnet_weights = torch.load(weight_str)
    resnet_model.load_state_dict(resnet_weights)
    
    return resnet_model

resnet_model = get_trained_resnet_model(
    layers=101,
    out_features=3,
    weight_str=model_load_root
)




criterion = nn.CrossEntropyLoss()
optimizer = Adam(resnet_model.parameters(), lr=0.001)

resnet_model.eval()
classifier = PyTorchClassifier(
    model=resnet_model,
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=3,
)

carlini_l2 = CarliniL2Method(classifier=classifier, batch_size=dataset_batch_size,max_iter=20, max_halving = 10, max_doubling=10 )

pil = transforms.ToPILImage()

if torch.cuda.is_available():
    resnet_model = resnet_model.cuda()

resnet_model.eval()
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="A simple Python script that takes command line arguments.")

    # add arguments
    parser.add_argument("-f", "--folderlocation", type=str, help="The location of the folder", required=True)
    parser.add_argument("-t", "--targetlocation", type=str, help="The target of the generated files", required=True)

    return parser.parse_args()
def main():
    # parse the command line arguments
    args = parse_args()
    dataset = CustomImageFolder(root=args.folderlocation,transform=img_transforms)
    loader = DataLoader(dataset,batch_size=dataset_batch_size,shuffle=True,num_workers=2,pin_memory=True)
    
    counter = 0
    for x, y in tqdm(loader):
        carlini_img = carlini_l2.generate(x=x.clone().numpy(), y=y.clone().numpy())
 

        for _,carlini in zip(y,carlini_img):
            
            carlini = carlini.transpose(1,2,0)
            carlini = np.clip(carlini,0,1,dtype=np.float32)
            
            
            image_bgr = cv2.cvtColor(carlini, cv2.COLOR_RGB2BGR,cv2.CV_32FC3)
            cv2.imwrite(
                img=image_bgr,
                filename=f"{args.targetlocation}/{counter}_carlini.tiff",
                params=[cv2.IMWRITE_TIFF_COMPRESSION, 1],
            )
            counter +=1


if __name__ == "__main__":
    main()


