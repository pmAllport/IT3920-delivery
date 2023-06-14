import os
import random
import argparse
import torch
import numpy as np, cv2
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from easydict import EasyDict
from dotenv import load_dotenv
load_dotenv()
project_root = os.getenv("PROJECT_ROOT")

pl.seed_everything(42, workers=True)

class DetectionImageFolderWithClass(datasets.ImageFolder):
    def __init__(self, root,img_transforms=None, target_transform=None,
                 loader=datasets.folder.default_loader, is_adversarial=None,should_return_class=False, should_return_attack_type=False):
        super(DetectionImageFolderWithClass, self).__init__(root, transform=img_transforms,
                                                target_transform=target_transform,
                                                loader=loader)
        self.is_adversarial = is_adversarial
        self.should_return_class = should_return_class
        self.should_return_attack_type = should_return_attack_type
        self.attack_mapping = {"apgd": 0, "carlini": 1, "fgsm": 2, "shadow": 3, "benign":4}
        self.classes = self._find_classes()

    def _find_classes(self):
        classes = []
        for _, class_idx in self.imgs:
            if class_idx not in classes:
                classes.append(class_idx)
        classes.sort()
        return classes
    

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        
        if img_path.endswith(".tiff"):
            loaded_tiff = cv2.imread(img_path,flags=-1,)
            rgb_tiff = cv2.cvtColor(loaded_tiff, cv2.COLOR_BGR2RGB)
            transposed_tiff = np.transpose(rgb_tiff,(2,0,1)).copy()
            img = torch.from_numpy(transposed_tiff)
        else:
            loaded_img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)
            transposed_rgb_img = np.transpose(rgb_img,(2,0,1)).copy() / 255.0
            img = torch.from_numpy(transposed_rgb_img).float()
 

        label = 1 if self.is_adversarial else 0

        if self.transform is not None:
            img = self.transform(img)


        if self.should_return_class:
            class_idx = self.imgs[index][1]
            if self.should_return_attack_type:
                if self.is_adversarial:
                    ## Retrieves the attack type, based on the img name
                    attack_type = [val for key,val in self.attack_mapping.items() if key in img_path][0]
                    return img, label, class_idx, attack_type
                    
                else:
                    return img, label, class_idx, self.attack_mapping["benign"]
            else:
                return img, label, class_idx
        else:
            return img, label


## TVT = Train,Val,Test
## This dataset is used when one wants to generate a dataset in which the labels either return if it is adversarial, and false if it is not adversarial.
## Overrides the data of the labels themselves.
def gen_tvt_detection_dataset(data_root, img_transforms, is_adversarial,train_root="train",val_root="val",test_root="test",should_return_attack_type=False,should_return_class=False):
    
    train = gen_detection_dataset(data_root=f'{data_root}/{train_root}',img_transforms=img_transforms,is_adversarial=is_adversarial,should_return_attack_type=should_return_attack_type,should_return_class=should_return_class)
    val = gen_detection_dataset(data_root=f'{data_root}/{val_root}',img_transforms=img_transforms,is_adversarial=is_adversarial,should_return_attack_type=should_return_attack_type,should_return_class=should_return_class)
    test = gen_detection_dataset(data_root=f'{data_root}/{test_root}',img_transforms=img_transforms,is_adversarial=is_adversarial,should_return_attack_type=should_return_attack_type,should_return_class=should_return_class)

    
    return train, val, test




## Returns a detection easydict, containing train, validation and test datasets as well as subsets of these three
## In this dict, the classification labels are overwritten, where the label indicates instead if the image is benign or adversarial.
def gen_detection_tvt_dict(data_root, img_transforms, batch_size, should_shuffle_datasets, is_adversarial, should_return_attack_type=False,should_return_class=False):
    
    
    train, val, test = gen_tvt_detection_dataset(
        data_root=data_root,
        img_transforms=img_transforms,
        is_adversarial=is_adversarial,
        should_return_class=should_return_class,
        should_return_attack_type= should_return_attack_type
    )
    

    loaders = {
        
        "train":DataLoader(train,batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
        "val":DataLoader(val,batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
        "test":DataLoader(test,batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
        

    }
    
    return EasyDict(
        
        train=loaders["train"],
        val=loaders["val"],
        test=loaders["test"],
        

        
        )


def concat_dataset_dicts(dict_1,dict_2, dict_str):
    return torch.utils.data.ConcatDataset([dict_1[dict_str].dataset, dict_2[dict_str].dataset])

def truncate_dataset(dataset, new_length, original_length):
    ## We need to set the seed or else the results would vary after each run.
    random.seed(42)
    random_index_list = random.sample(range(0,original_length),new_length, )

    return torch.utils.data.Subset(dataset, random_index_list)

def truncate_and_concat_dicts(dict_1, dict_2, dict_str):

    dataset_1 = dict_1[dict_str].dataset
    dataset_2 = dict_2[dict_str].dataset
    len_dataset1 = len(dataset_1)
    len_dataset2 = len(dataset_2)

    if len_dataset1 > len_dataset2:
        dataset_1 = truncate_dataset(dataset_1, len_dataset2, len_dataset1)
    elif len_dataset2 > len_dataset1:
        dataset_2 = truncate_dataset(dataset_2, len_dataset1, len_dataset2)

    return torch.utils.data.ConcatDataset([dataset_1, dataset_2])

    
    
    
## Generates a detection dataset, containing both benign and adversarial images 
def gen_mixed_detection_tvt_dict(ben_root, adv_root, img_transforms, batch_size,should_shuffle_datasets,num_workers=8,pin_memory=True, should_only_transform_benign=False, should_truncate_datasets=False, should_return_class=False, should_return_attack_type=False):

    concat_function = truncate_and_concat_dicts if should_truncate_datasets else concat_dataset_dicts
    
    
    ben_dict = gen_detection_tvt_dict(
        data_root=ben_root,
        img_transforms=img_transforms,
        batch_size=batch_size,
        should_shuffle_datasets=should_shuffle_datasets,
        is_adversarial=False,
        should_return_class=should_return_class,
        should_return_attack_type=should_return_attack_type,
    )
    
    adv_dict = gen_detection_tvt_dict(
        data_root=adv_root,
        img_transforms=img_transforms,
        batch_size=batch_size,
        should_shuffle_datasets=should_shuffle_datasets,
        is_adversarial=True,
        should_return_class=should_return_class,
        should_return_attack_type=should_return_attack_type
    )
    
    loaders = {
        
        "train":DataLoader(concat_function(ben_dict,adv_dict,"train"),batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=num_workers,pin_memory=pin_memory),
        "val":DataLoader(concat_function(ben_dict,adv_dict,"val"),batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory),
        "test":DataLoader(concat_function(ben_dict,adv_dict,"test"),batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory),

    }
    
    return EasyDict(
        
        train=loaders["train"],
        val=loaders["val"],
        test=loaders["test"],
        
        )


def gen_detection_dataset(data_root, img_transforms, is_adversarial, should_return_attack_type=False,should_return_class=False):
    
    return DetectionImageFolderWithClass(root=f'{data_root}',img_transforms=img_transforms,is_adversarial=is_adversarial, should_return_attack_type=should_return_attack_type,should_return_class=should_return_class)






benign_root = f"{project_root}/datasets/final_datasets/benign/full_size"
adv_root = f"{project_root}/datasets/final_datasets/adversarial"



## Needs to be downscaled to support resnets dense layers, which require 224,224 image sizes
resnet_im_size = (224,224)
dataset_batch_size=32



def normalize_fft(tensor, eps=1e-8):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min + eps)
    return tensor



img_transforms = transforms.Compose([
    transforms.Lambda(lambda x:  F.interpolate(x.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False, antialias=True).squeeze(0)),
    transforms.Lambda(lambda x: torch.flatten(torch.abs(torch.fft.fft2(x)).real)),
    transforms.Lambda(normalize_fft)
])



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_accuracy(output, target):
    
    output = (output >= 0.5).int()
    target = (target == 1.0).int()
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()


class LogisticRegressionPL(pl.LightningModule):
    def __init__(self, input_features,lr=1e-03):
        super(LogisticRegressionPL, self).__init__()
        self.linear = torch.nn.Linear(input_features, 1, dtype=torch.float32)
        self.lr = lr
    def forward(self, x):
        x = x.view(x.size(0), -1)
        preds = self.linear(x)
        return preds  

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        preds = preds.squeeze(1)
        loss = nn.BCEWithLogitsLoss()(preds, y.to(torch.float32))  # Change to binary_cross_entropy_with_logits
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        preds = self(x)
        preds = preds.squeeze(1)
        loss = nn.BCEWithLogitsLoss()(preds, y.to(torch.float32))  # Change to binary_cross_entropy_with_logits
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        preds = preds.squeeze(1)
        loss = nn.BCEWithLogitsLoss()(preds, y.to(torch.float32))   # Change to binary_cross_entropy_with_logits
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


def parse_args():
    parser = argparse.ArgumentParser(description="A simple Python script that takes command line arguments.")

    # add arguments
    parser.add_argument("-e", "--epochs", type=str, help="The number of epochs", required=True)
    parser.add_argument("-lr", "--learningrate", type=float, help="The learning rate of the model", required=True)

    return parser.parse_args()
def main():
    # parse the command line arguments
    args = parse_args()
    epochs = int(args.epochs)
    lr = args.learningrate
    ## As it is really easy to overwrite the model during training,
    ## the saveroot points to a subdirectory of the weights, rather than the models themselves
    ## i.e. save_root = f"{project_root}/weights/recovery/run_test"
    ## instead of
    ## save_root = f"{project_root}/weights/recovery"
    ## This is done to avoid spectral.pt from being overwritten by accident
    save_root = f"{project_root}/weights/recovery/run_test"
    os.makedirs(save_root,exist_ok=True)
    mixed_data = gen_mixed_detection_tvt_dict(
        ben_root=benign_root,
        adv_root=adv_root,
        img_transforms=img_transforms,
        batch_size=dataset_batch_size,
        should_shuffle_datasets=True,
        should_truncate_datasets=True,
        num_workers=6
    )

    callbacks = [pl.callbacks.EarlyStopping(monitor="val_loss",patience=2,mode="min")]
    
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=save_root,
        filename="sad-model-{epoch}-{val_loss:.2f}-{val_acc:0.2f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
    ))

    trainer = pl.Trainer(
    accelerator="gpu",
    strategy="auto",
    max_epochs=epochs,
    precision="16",
    callbacks= callbacks,
    )


    detector = LogisticRegressionPL(3*224*224, lr=lr)

    trainer.fit(detector,train_dataloaders=mixed_data.train,val_dataloaders=mixed_data.val)


    trainer.test(detector,dataloaders=mixed_data.test)
        
    torch.save(trainer.model.state_dict(), f"{save_root}/spectral.pt")


if __name__ == "__main__":
    main()