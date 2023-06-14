
from torchvision import datasets
from random import sample
from torch.optim import SGD, Adam
from torchmetrics import Accuracy
import torchvision.models as models
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms
import os
import torch
from torch.utils.data import Dataset
import torch
import numpy as np, cv2
from PIL import Image
import os
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, make_dataset
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

class SubsetImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, num_per_class=5, is_adversarial=None):
        super(SubsetImageFolder, self).__init__(root, transform=transform,
                                                target_transform=target_transform,
                                                loader=loader)
        self.num_per_class = num_per_class
        self.is_adversarial = is_adversarial
        self.indices = self._select_indices()

    def _select_indices(self):
        indices = []
        for class_idx in range(len(self.classes)):
            class_indices = [idx for idx in range(len(self.imgs))
                             if self.imgs[idx][1] == class_idx]
            if len(class_indices) <= self.num_per_class:
                indices += class_indices
            else:
                indices += sample(class_indices, self.num_per_class)
        return indices

    def __getitem__(self, index):
        original_index = self.indices[index]
        img, label = super(SubsetImageFolder, self).__getitem__(original_index)
        if self.is_adversarial is None:
            return img, label
        else:
            label = 1 if self.is_adversarial else 0
            return img, label

    def __len__(self):
        return len(self.indices)
    


class DetectionImageFolderWithClass(datasets.ImageFolder):
    def __init__(self, root,img_transforms=None, target_transform=None,
                 loader=datasets.folder.default_loader,num_per_class=None, is_adversarial=None,should_return_class=False, should_return_attack_type=False):
        super(DetectionImageFolderWithClass, self).__init__(root, transform=img_transforms,
                                                target_transform=target_transform,
                                                loader=loader)
        self.is_adversarial = is_adversarial
        self.num_per_class = num_per_class
        self.should_return_class = should_return_class
        self.should_return_attack_type = should_return_attack_type
        self.attack_mapping = {"apgd": 0, "carlini": 1, "fgsm": 2, "shadow": 3, "benign":4}
        self.classes = self._find_classes()
        self.indices = self._select_indices()

    def _find_classes(self):
        classes = []
        for _, class_idx in self.imgs:
            if class_idx not in classes:
                classes.append(class_idx)
        classes.sort()
        return classes

    
    def _select_indices(self):
        indices = []

        for class_idx in range(len(self.classes)):
            class_indices = [idx for idx in range(len(self.imgs))
                             if self.imgs[idx][1] == class_idx]
            if self.num_per_class is None or len(class_indices) <= self.num_per_class:
                indices += class_indices
            else:
                indices += sample(class_indices, self.num_per_class)
            
            indices += class_indices
        return indices

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
        
    
       
class AdversarialImageAttackLabelDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        self.attack_mapping = {"apgd": 0, "carlini": 1, "fgsm": 2, "shadow": 3}
        
        self.samples = []
        for label_idx, class_folder in enumerate(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_folder)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                attack_name = image_name.split("_")[1].split(".")[0]
                self.samples.append((image_path, label_idx, self.attack_mapping[attack_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, attack_type = self.samples[index]
        
        
        if image_path.endswith(".tiff"):
            loaded_tiff = cv2.imread(image_path,flags=-1,)
            rgb_tiff = cv2.cvtColor(loaded_tiff, cv2.COLOR_BGR2RGB)
            transposed_tiff = np.transpose(rgb_tiff,(2,0,1))
            image = torch.from_numpy(transposed_tiff)
        else:
            image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, attack_type

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        super(CustomImageFolder, self).__init__()

        self.root = root
        self.transform = transform
        self.target_transform = target_transform


        samples = make_dataset(self.root, class_to_idx=self._find_classes(self.root), extensions=IMG_EXTENSIONS)
        if len(samples) == 0:
            raise RuntimeError(f"Found 0 files in subfolders of: {self.root}\nSupported extensions are: {', '.join(IMG_EXTENSIONS)}")

        self.loader = loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        if path.endswith(".tiff"):
            loaded_tiff = cv2.imread(path,flags=-1,)
            rgb_tiff = cv2.cvtColor(loaded_tiff, cv2.COLOR_BGR2RGB)
            transposed_tiff = np.transpose(rgb_tiff,(2,0,1))
            sample = torch.from_numpy(transposed_tiff)
        else:
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


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
        return {
            "optimizer":optimizer,
            "gradient_clip_val":1.0,
                }

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

## Code taken from https://github.com/Stevellen/ResNet-Lightning and modified to fit the use
class NVPResnetClassifierDualtransform(pl.LightningModule):
    optimizers = {"adam": Adam, "sgd": SGD}
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        custom_channel_value=None,
        use_transform=None
    ):
        super().__init__()
    
        self.lr = lr
        self.batch_size = batch_size
        self.custom_channel_value = custom_channel_value
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.nvp_functions = {
                            "jpg_minpool_jpg": transforms.Compose([JPEGCompressionTransform(),MaxPoolTransform(), JPEGCompressionTransform()]),
                            "maxpool": MaxPoolTransform(),
                            "minpool": MinPoolTransform(),
                            "minpool_jpg" : transforms.Compose([MaxPoolTransform(), JPEGCompressionTransform()]),
                            }
        
        self.nvp_funcs = [list(self.nvp_functions.items())[i][1] for i in list(use_transform)] if use_transform is not None else self.nvp_functions
        self.individual_transform = use_transform

        self.optimizer = self.optimizers[optimizer]
        # instantiate loss criterion
        self.loss_fn = nn.BCEWithLogitsLoss()
        # create accuracy metric
        self.acc = Accuracy(task="binary")
        # Using a pretrained ResNet backbone
        self.resnet_model = models.resnet101()
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, 1)
        
        self.use_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.pred_tensor = torch.cuda.HalfTensor if  torch.cuda.is_available() else torch.HalfTensor 
        
        if self.custom_channel_value is not None:
            self.resnet_model.conv1 = nn.Conv2d(self.custom_channel_value, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    
    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer":optimizer,"gradient_clip_val":1.0}

    def _step(self, batch):
        x, y = batch
        train_preds = []
        for nvp_func in self.nvp_funcs:
            nvp_imgs = nvp_func(x).to(self.use_device)
            train_preds.append(nvp_imgs)
        stacked = torch.concat(train_preds,dim=1)
        preds = self(stacked.type(self.pred_tensor))
        loss = nn.BCEWithLogitsLoss()(preds.squeeze(1), y.to(torch.float32))

        preds = preds.flatten()
        target_y = y.float()

        loss = self.loss_fn(preds, target_y)
        acc = self.acc(preds, target_y)
        return loss, acc

    def train_dataloader(self):
        return self.train_dataloader

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def val_dataloader(self):
        return self.val_dataloader

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_dataloader(self):
        return self.test_dataloader

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
        
def process_maxpool_tensor(t, pool):
    return pool(t)

def process_minpool_tensor(t, pool):
    return -pool(-t)
class MaxPoolTransform(nn.Module):
    def __init__(self):
        super(MaxPoolTransform, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=(5, 5), stride=2)

    def forward(self, tensor):
        pool = nn.MaxPool2d(kernel_size=(5, 5), stride=2)
        pool = pool.to(tensor.device)  # Ensure the pooling layer is on the same device as the input tensor

        # Create a list of tensors to process in parallel
        batch_size = tensor.shape[0]
        tensors_to_process = []
        for i in range(batch_size):
            tensors_to_process.append((tensor[i:i+1, :, :, :], pool))

        # Use a ThreadPoolExecutor to apply the function to each tensor in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda x: process_maxpool_tensor(*x), tensors_to_process))

        max_pooled_output = torch.cat(results, dim=0)
        upscaled_tensors = nn.functional.interpolate(
            input=max_pooled_output,
            size=(224, 224),
            mode='bicubic',
            align_corners=False
        )

        return upscaled_tensors

    
class MinPoolTransform(nn.Module):
    def __init__(self):
        super(MinPoolTransform, self).__init__()
        self.min_pool = nn.MaxPool2d(kernel_size=(5, 5), stride=2)

    def forward(self, tensor):
        pool = nn.MaxPool2d(kernel_size=(5, 5), stride=2)
        pool = pool.to(tensor.device)  # Ensure the pooling layer is on the same device as the input tensor

        # Create a list of tensors to process in parallel
        batch_size = tensor.shape[0]
        tensors_to_process = []
        for i in range(batch_size):
            tensors_to_process.append((tensor[i:i+1, :, :, :], pool))

        # Use a ThreadPoolExecutor to apply the function to each tensor in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda x: process_minpool_tensor(*x), tensors_to_process))

        max_pooled_output = torch.cat(results, dim=0)
        upscaled_tensors = nn.functional.interpolate(
            input=max_pooled_output,
            size=(224, 224),
            mode='bicubic',
            align_corners=False
        )

        return upscaled_tensors

    


class JPEGCompressionTransform(nn.Module):
    def __init__(self, quality=90):
        super(JPEGCompressionTransform, self).__init__()
        self.quality = quality

    def compress_image(self, img):
        img = img.mul(255).byte()
        img = img.cpu().numpy().transpose((1, 2, 0)) # Convert to HWC

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, buffer = cv2.imencode('.jpg', img, encode_param)

        compressed_img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        compressed_img = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        compressed_img = compressed_img.transpose((2, 0, 1)) # Convert to CHW

        return torch.from_numpy(compressed_img)

    def forward(self, tensor):
        batch_size = tensor.size(0)
        compressed_batch = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            compressed_batch = list(executor.map(self.compress_image, [tensor[i] for i in range(batch_size)]))

        return torch.stack(compressed_batch, dim=0).float() / 255.0
    
## Code taken from https://github.com/Stevellen/ResNet-Lightning and modified to fit the use
class NVPResnetClassifierIndiv(pl.LightningModule):
    optimizers = {"adam": Adam, "sgd": SGD}
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        custom_channel_value=None,
        use_transform=None
    ):
        super().__init__()
    
        self.lr = lr
        self.batch_size = batch_size
        self.custom_channel_value = custom_channel_value
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.nvp_functions = {
                            "jpg_minpool_jpg": transforms.Compose([JPEGCompressionTransform(),MaxPoolTransform(), JPEGCompressionTransform()]),
                            "maxpool": MaxPoolTransform(),
                            "minpool": MinPoolTransform(),
                            "minpool_jpg" : transforms.Compose([MaxPoolTransform(), JPEGCompressionTransform()]),
                            }
        
        self.nvp_funcs = list(self.nvp_functions.items())[int(use_transform)] if use_transform is not None else self.nvp_functions
        self.individual_transform = use_transform

        self.optimizer = self.optimizers[optimizer]
        # instantiate loss criterion
        self.loss_fn = nn.BCEWithLogitsLoss()
        # create accuracy metric
        self.acc = Accuracy(task="binary")
        # Using a pretrained ResNet backbone
        self.resnet_model = models.resnet101()
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, 1)
        
        self.use_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.pred_tensor = torch.cuda.HalfTensor if  torch.cuda.is_available() else torch.HalfTensor 
        
        if self.custom_channel_value is not None:
            self.resnet_model.conv1 = nn.Conv2d(self.custom_channel_value, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    
    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer":optimizer,"gradient_clip_val":1.0}

    def _step(self, batch):
        x, y = batch

        nvp_imgs = self.nvp_funcs[1](x).to(self.use_device)
        preds = self(nvp_imgs.type(self.pred_tensor)).squeeze(1)
        loss = nn.BCEWithLogitsLoss()(preds, y.to(torch.float32))

        preds = preds.flatten()
        target_y = y.float()

        loss = self.loss_fn(preds, target_y)
        acc = self.acc(preds, target_y)
        return loss, acc

    def train_dataloader(self):
        return self.train_dataloader

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def val_dataloader(self):
        return self.val_dataloader

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_dataloader(self):
        return self.test_dataloader

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
    
nvp_functions = {
    "jpg_minpool_jpg": transforms.Compose([JPEGCompressionTransform(),MaxPoolTransform(), JPEGCompressionTransform()]),
    "maxpool": MaxPoolTransform(),
    "minpool": MinPoolTransform(),
    "minpool_jpg" : transforms.Compose([MaxPoolTransform(), JPEGCompressionTransform()]),
    }   

## Code taken from https://github.com/Stevellen/ResNet-Lightning and modified to fit the use    
class NVPResnetClassifier(pl.LightningModule):
    optimizers = {"adam": Adam, "sgd": SGD}

    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        custom_channel_value=None
    ):
        super().__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.custom_channel_value = custom_channel_value
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = self.optimizers[optimizer]
        # instantiate loss criterion
        self.loss_fn = nn.BCEWithLogitsLoss()
        # create accuracy metric
        self.acc = Accuracy(task="binary")
        # Using a pretrained ResNet backbone
        self.resnet_model = models.resnet101()
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, 1)
        
        self.use_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.pred_tensor = torch.cuda.HalfTensor if  torch.cuda.is_available() else torch.HalfTensor 
        
        if self.custom_channel_value is not None:
            self.resnet_model.conv1 = nn.Conv2d(self.custom_channel_value, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer":optimizer}

    def _step(self, batch):
        x, y = batch
        train_preds = []
        for _,nvp_func in nvp_functions.items():
            nvp_imgs = nvp_func(x).to(self.use_device)
            train_preds.append(nvp_imgs)
        stacked = torch.concat(train_preds,dim=1)
        preds = self(stacked).squeeze(1)
        loss = nn.BCEWithLogitsLoss()(preds, y.to(torch.float32))

        preds = preds.flatten()
        target_y = y.float()

        loss = self.loss_fn(preds, target_y)
        acc = self.acc(preds, target_y)
        return loss, acc

    def train_dataloader(self):
        return self.train_dataloader

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def val_dataloader(self):
        return self.val_dataloader

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_dataloader(self):
        return self.test_dataloader

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
        