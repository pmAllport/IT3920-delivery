from easydict import EasyDict
from torchvision.transforms import Compose, Resize, ToTensor
import torchvision.models as models
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import ctypes
import random
from classes import *

def test_benign_model(model, test_dataloader,device):
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in test_dataloader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            _, y_pred = model(x).max(1)
            report.nb_test += y.size(0)
            report.correct += y_pred.eq(y).sum().item()
    print(
        "test acc on examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )


def test_torch_detection_model(model, mixed_dataloader,threshold, device):
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in mixed_dataloader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            preds = model(x)
            sig_preds = (torch.sigmoid(preds)>threshold).int()
            report.nb_test += y.size(0)
            for p, gt in zip(sig_preds,y):
                report.correct += p.eq(gt.item()).sum().item()
    print(
        "test acc on mixed examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )
            
def get_label_arr_and_map(torch_features):
    if torch_features == 4:
        int_labels = [
            "pedestrian",
            "static_heavy",
            "static_light",
            "vehicle",
        ]

    elif torch_features == 23:
        int_labels = [
            "animal",
            "human.pedestrian.adult",
            "human.pedestrian.child",
            "human.pedestrian.construction_worker",
            "human.pedestrian.personal_mobility",
            "human.pedestrian.police_officer",
            "human.pedestrian.stroller",
            "human.pedestrian.wheelchair",
            "movable_object.barrier",
            "movable_object.debris",
            "movable_object.pushable_pullable",
            "movable_object.trafficcone",
            "static_object.bicycle_rack",
            "vehicle.bicycle",
            "vehicle.bus.bendy",
            "vehicle.bus.rigid",
            "vehicle.car",
            "vehicle.construction",
            "vehicle.emergency.ambulance",
            "vehicle.emergency.police",
            "vehicle.motorcycle",
            "vehicle.trailer",
            "vehicle.truck"
            ]
    elif torch_features == 3:
        int_labels = [
            "inanimate",
            "pedestrian",
            "vehicle",
        ]
    else:
        raise Exception(f"No labels found for torch_features equal to {torch_features}")
    
    label_map = {label: i for i, label in enumerate(int_labels)}
    
    if int_labels == None:
        raise Exception(f"No valid int_labels exist for torch_features equal to {torch_features}.")
    else:
        return int_labels, label_map

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

## TVT = Train,Val,Test
## Generates a Imagefolder, pointing only to a subset of the images in the folder. Primarily used for testing of solutions.
def gen_tvt_subset(data_root, transforms, train_root="train",val_root="val",test_root="test", is_adversarial=None):
    
    train = SubsetImageFolder(root=f'{data_root}/{train_root}',  transform=transforms,is_adversarial=is_adversarial)
    val = SubsetImageFolder(root=f'{data_root}/{val_root}',  transform=transforms,is_adversarial=is_adversarial)
    test = SubsetImageFolder(root=f'{data_root}/{test_root}', transform=transforms,is_adversarial=is_adversarial)
    
    return train, val, test




## TVT = Train,Val,Test
## Generates a classification dataset. Returns a Train, validation and test Imagefolder.
def gen_classification_tvt_dataset(data_root, transforms, should_return_attack_labels=False, should_return_path=False):
    dataset_class = AdversarialImageAttackLabelDataset if should_return_attack_labels else CustomImageFolder
    
    train = dataset_class(root=f'{data_root}train',transform=transforms)
    val = dataset_class(root=f'{data_root}val', transform=transforms)
    test = dataset_class(root=f'{data_root}test', transform=transforms)
    
    return train, val, test
    




## TVT = Train,Val,Test
## This dataset is used when one wants to generate a dataset in which the labels either return if it is adversarial, and false if it is not adversarial.
## Overrides the data of the labels themselves.
def gen_tvt_detection_dataset(data_root, img_transforms, is_adversarial,train_root="train",val_root="val",test_root="test",should_return_attack_type=False,should_return_class=False, num_per_class=None):
    
    train = gen_detection_dataset(data_root=f'{data_root}/{train_root}',img_transforms=img_transforms,is_adversarial=is_adversarial,should_return_attack_type=should_return_attack_type,should_return_class=should_return_class,num_per_class=num_per_class)
    val = gen_detection_dataset(data_root=f'{data_root}/{val_root}',img_transforms=img_transforms,is_adversarial=is_adversarial,should_return_attack_type=should_return_attack_type,should_return_class=should_return_class,num_per_class=num_per_class)
    test = gen_detection_dataset(data_root=f'{data_root}/{test_root}',img_transforms=img_transforms,is_adversarial=is_adversarial,should_return_attack_type=should_return_attack_type,should_return_class=should_return_class,num_per_class=num_per_class)

    
    return train, val, test



## Generates a default classification easydict, containing Train, validation and test as well as subsets for all of these datasets.
def gen_classification_tvt_dict(data_root, transforms, batch_size, should_shuffle_datasets, should_return_attack_labels=False, target_is_tiff=False, should_return_path=False):
    
    train, val, test = gen_classification_tvt_dataset(
        data_root=data_root,
        transforms=transforms,
        should_return_attack_labels=should_return_attack_labels,
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
    
## This returns a mixed classification dict. One should have no reason to use this, but it is here just in case.
def gen_mixed_classificiation_tvt_dict(ben_root, adv_root, transforms, batch_size, num_per_class,should_shuffle_datasets):
    
    
    ben_dict = gen_classification_tvt_dict(
        data_root=ben_root,
        transforms=transforms,
        batch_size=batch_size,
        num_per_class=num_per_class,
        should_shuffle_datasets=should_shuffle_datasets
    )
    
    adv_dict = gen_classification_tvt_dict(
        data_root=adv_root,
        transforms=transforms,
        batch_size=batch_size,
        num_per_class=num_per_class,
        should_shuffle_datasets=should_shuffle_datasets
    )
    
    loaders = {
        
        "train":DataLoader(concat_dataset_dicts(ben_dict,adv_dict,"train"),batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
        "val":DataLoader(concat_dataset_dicts(ben_dict,adv_dict,"val"),batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
        "test":DataLoader(concat_dataset_dicts(ben_dict,adv_dict,"test"),batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
        
        "train_subset":DataLoader(concat_dataset_dicts(ben_dict,adv_dict,"train_subset"),batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
        "val_subset":DataLoader(concat_dataset_dicts(ben_dict,adv_dict,"val_subset"),batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
        "test_subset":DataLoader(concat_dataset_dicts(ben_dict,adv_dict,"test_subset"),batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
    }
    
    return EasyDict(
        
        train=loaders["train"],
        val=loaders["val"],
        test=loaders["test"],
        
        train_subset=loaders["train_subset"],
        val_subset=loaders["val_subset"],
        test_subset=loaders["test_subset"],
        
        )
    
    
    
## Generates a detection dataset, containing both benign and adversarial images 
def gen_mixed_detection_tvt_dict(ben_root, adv_root, img_transforms, batch_size,should_shuffle_datasets,num_workers=8,pin_memory=True, should_only_transform_benign=False, should_truncate_datasets=False, should_return_class=False, should_return_attack_type=False):
    # If the adversarial images are already normalized, then we cannot normalize them again.
    adv_transforms = Compose([Resize((224, 224)),ToTensor(),]) if should_only_transform_benign else transforms
    
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


def gen_detection_dataset(data_root, img_transforms, is_adversarial, should_return_attack_type=False,should_return_class=False, num_per_class=None):
    
    return DetectionImageFolderWithClass(root=f'{data_root}',img_transforms=img_transforms,is_adversarial=is_adversarial, should_return_attack_type=should_return_attack_type,should_return_class=should_return_class, num_per_class=num_per_class)



def gen_samples_subset(data_root, num_per_class, transforms, is_adversarial=None):
    
    return SubsetImageFolder(root=f'{data_root}', num_per_class=num_per_class, transform=transforms,is_adversarial=is_adversarial)

def gen_detection_samples_dict(data_root, transforms, num_per_class, batch_size, should_shuffle_datasets, is_adversarial):
    
    
    samples = gen_detection_dataset(
        data_root=data_root,
        transforms=transforms,
        is_adversarial=is_adversarial
    )
    
    samples_subset = gen_samples_subset(
        data_root=data_root,
        transforms=transforms,
        num_per_class=num_per_class,
        is_adversarial=is_adversarial
    )
    loaders = {
        
        "samples":DataLoader(samples,batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
        
        "samples_subset":DataLoader(samples_subset,batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=8,pin_memory=True),
    }
    
    return EasyDict(
        
        samples=loaders["samples"],
        
        samples_subset=loaders["samples_subset"],
        
        )

def gen_nvp_detection_dataset(data_root, img_transforms, voter_transforms, is_adversarial, should_return_class=False, num_per_class=None):
        
        
        # This may seem like a bit of black magic, but what this essencially does is:
        # - Take the train, val and test dataset from each of the "voter transforms" (e.g. minpool/maxpool/ etc.)
        # - Transpose the array, swapping the first and second axis. E.g. -> 4x3x96x2 to 3x4x96x2.
        # Each of the elements in the first axis, refer to the training set for each of the voter transformed images. E.g. training set for minpool, training set for maxpool etc.
        # - It then concats each of these datasets, so that the training set for e.g. minpool, maxpool etc. are merged into one LARGE training, validation and test set.
        # In this example, the returned example has a shape of 3x96x2, where the elements in the first axes are train, val and test
        
        generate_detection_dataset_func = gen_tvt_subset if num_per_class is not None else gen_tvt_detection_dataset
        
        tvt_dataset = [
                            generate_detection_dataset_func(
                            data_root=f"{data_root}/{voter_transform}",
                            transforms=img_transforms,
                            is_adversarial=is_adversarial,
                            should_return_class=should_return_class,
                            num_per_class=num_per_class,
                            train_root="train",
                            val_root="val",
                            test_root="test") for voter_transform in voter_transforms
                        ]
        
        transposed = [[x[i] for x in tvt_dataset] for i in range(0,len(["train","val","test"]))]
        
        
        concat = [
            torch.utils.data.ConcatDataset([x]) for x in transposed
        ]
        
        return concat
    
 
        
def gen_nvp_detection_tvt_dict(data_root, num_per_class, img_transforms, voter_transforms, is_adversarial,batch_size, should_shuffle_datasets, should_return_class):
    
    dataset = gen_nvp_detection_dataset(
        data_root=data_root,
        img_transforms=img_transforms,
        voter_transforms=voter_transforms,
        is_adversarial=is_adversarial,
        num_per_class=None,
        should_return_class=should_return_class
    )
    
    
    
    subset = gen_nvp_detection_dataset(
        data_root=data_root,
        img_transforms=img_transforms,
        voter_transforms=voter_transforms,
        is_adversarial=is_adversarial,
        num_per_class=num_per_class,
        should_return_class=should_return_class
    )
    loaders = {
        
        "train":DataLoader(dataset[0],batch_size=batch_size,shuffle=should_shuffle_datasets),
        "val":DataLoader(dataset[1],batch_size=batch_size,shuffle=should_shuffle_datasets),
        "test":DataLoader(dataset[2],batch_size=batch_size,shuffle=should_shuffle_datasets),
        
        "train_subset":DataLoader(subset[0],batch_size=batch_size,shuffle=should_shuffle_datasets),
        "val_subset":DataLoader(subset[1],batch_size=batch_size,shuffle=should_shuffle_datasets),
        "test_subset":DataLoader(subset[2],batch_size=batch_size,shuffle=should_shuffle_datasets),
        }
    
    return EasyDict(
        
        train=loaders["train"],
        val=loaders["val"],
        test=loaders["test"],
        
        train_subset=loaders["train_subset"],
        val_subset=loaders["val_subset"],
        test_subset=loaders["test_subset"],
        
        )
        
def gen_mixed_nvp_detection_tvt(ben_root, adv_root, img_transforms, batch_size, num_per_class,voter_transforms,should_shuffle_datasets,num_workers=8,pin_memory=True):
    
    
    ben_dict = gen_nvp_detection_tvt_dict(
        data_root=ben_root,
        num_per_class=num_per_class,
        is_adversarial=False,
        img_transforms=img_transforms,
        voter_transforms=voter_transforms,
        should_shuffle_datasets=should_shuffle_datasets,
        batch_size=batch_size,
    )
    
    adv_dict = gen_nvp_detection_tvt_dict(
        data_root=adv_root,
        num_per_class=num_per_class,
        is_adversarial=True,
        img_transforms=img_transforms,
        voter_transforms=voter_transforms,
        should_shuffle_datasets=should_shuffle_datasets,
        batch_size=batch_size,
    )
    
    def concat_datasets(dict_1,dict_2, dict_str):
        return torch.utils.data.ConcatDataset([*dict_1[dict_str].dataset, *dict_2[dict_str].dataset])
  
    loaders = {
        
        "train":DataLoader(concat_datasets(ben_dict,adv_dict,"train"),batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=num_workers,pin_memory=pin_memory),
        "val":DataLoader(concat_datasets(ben_dict,adv_dict,"val"),batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory),
        "test":DataLoader(concat_datasets(ben_dict,adv_dict,"test"),batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory),
        
        "train_subset":DataLoader(concat_datasets(ben_dict,adv_dict,"train_subset"),batch_size=batch_size,shuffle=should_shuffle_datasets,num_workers=num_workers,pin_memory=pin_memory),
        "val_subset":DataLoader(concat_datasets(ben_dict,adv_dict,"val_subset"),batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory),
        "test_subset":DataLoader(concat_datasets(ben_dict,adv_dict,"test_subset"),batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory),
    }
    
    return EasyDict(
        
        train=loaders["train"],
        val=loaders["val"],
        test=loaders["test"],
        
        train_subset=loaders["train_subset"],
        val_subset=loaders["val_subset"],
        test_subset=loaders["test_subset"],
        
        )
        
        
## We need this collate function in the "gen_multi_root_nvp_characteristics_dict" function, or else the dataloader will return a the data in the shape with
## batch size of 1. E.g. return as (1x64x12x224x224) instead of (64x12x224x224)
def custom_collate(batch):
    images, is_adv = zip(*batch)
    images = torch.stack(images, dim=0).squeeze(0)
    is_adv = torch.stack(is_adv, dim=0).squeeze(0)
    if images.dim() == 5:
        images = images.view(-1,12,224,224)
        is_adv = is_adv.view(-1)
    return images, is_adv 

def custom_collate_with_classes(batch):
    images, is_adv, labels  = zip(*batch)
    images = torch.stack(images, dim=0).squeeze(0)
    labels = torch.stack(labels, dim=0).squeeze(0)
    is_adv = torch.stack(is_adv, dim=0).squeeze(0)
    if images.dim() == 5:
        images = images.view(-1,12,224,224)
        is_adv = is_adv.view(-1)
        labels = labels.view(-1)
    return images, is_adv, labels

    
    
def gen_easydict():
    return EasyDict(
        nb_test=0, 
        correct=0, 
        false_positives=0, 
        false_negatives=0,
        true_positives=0,
        true_negatives=0,
        nb_test_pedestrian=0, 
        correct_pedestrian=0, 
        false_positives_pedestrian=0, 
        false_negatives_pedestrian=0,
        true_positives_pedestrian=0,
        true_negatives_pedestrian=0,
        nb_test_inanimate=0, 
        correct_inanimate=0, 
        false_positives_inanimate=0, 
        false_negatives_inanimate=0,
        true_positives_inanimate=0,
        true_negatives_inanimate=0,
        nb_test_vehicle=0, 
        correct_vehicle=0, 
        false_positives_vehicle=0, 
        false_negatives_vehicle=0,
        true_positives_vehicle=0,
        true_negatives_vehicle=0,
        roc_auc=[],
        roc_auc_inanimate=[],
        roc_auc_pedestrian=[],
        roc_auc_vehicle=[]
    )
    
def inner_write_to_easydict(easydict, prediction, ground_truth, float_prediction, classification_string):
    number_string = f"nb_test{classification_string}"
    correct_string = f"correct{classification_string}"
    fp_string = f"false_positives{classification_string}"
    fn_string = f"false_negatives{classification_string}"
    tp_string = f"true_positives{classification_string}"
    tn_string = f"true_negatives{classification_string}"
    roc_auc_string = f"roc_auc{classification_string}"
    
    easydict[number_string] += 1
    if prediction == ground_truth:
        easydict[correct_string] += 1
        if prediction == 1:
            easydict[tp_string] += 1
        else:
            easydict[tn_string] += 1
    elif prediction == 1 and ground_truth == 0:
        easydict[fp_string] += 1
    elif prediction == 0 and ground_truth == 1:
        easydict[fn_string] += 1
    easydict[roc_auc_string].append((float_prediction,float(ground_truth)))

def write_to_easydict(easydict, prediction, ground_truth, float_prediction,classification=None):
    inner_write_to_easydict(easydict, prediction, ground_truth, float_prediction, "" )

    if classification:
        classification_str = f"_{str(classification)}"
        inner_write_to_easydict(easydict, prediction, ground_truth, float_prediction, classification_str,)

int_labels = ['inanimate', 'pedestrian', 'vehicle']



def print_easydict_results(easydict, class_str, has_classifications=False, uses_roc_auc=False, eps=1e-12):
    print(f"Printing results for {class_str}\n")
    print("-------------------------------")
    print(
        "test acc on mixed examples (%): {:.3f}".format(
            easydict.correct / easydict.nb_test * 100.0
        )
    )

    fpr = easydict.false_positives / (easydict.false_positives + easydict.true_negatives )
    print(
        "False positive rate on mixed examples (%): {:.3f}".format(
            fpr * 100.0
        )
    )

    fnr = easydict.false_negatives / (easydict.false_negatives + easydict.true_positives)
    print(
        "False negative rate on mixed examples (%): {:.3f}".format(
            fnr * 100.0
        )
    )
    ## Add small eps to prevent div by 0 without altering the data much
    precision = easydict.true_positives / (easydict.true_positives + easydict.false_positives)
    print(
        "Precision on mixed examples (%): {:.3f}".format(
            precision * 100.0
        )
    )
    ## Add small eps to prevent div by 0 without altering the data much
    recall = easydict.true_positives / (easydict.true_positives + easydict.false_negatives)
    print(
        "Recall on mixed examples (%): {:.3f}".format(
            recall * 100.0
        )
    )
    ## Add small eps to prevent div by 0 without altering the data much
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print(
        "F1-score on mixed examples (%): {:.3f}".format(
            f1_score * 100.0
        )
    )
    if uses_roc_auc:
        float_scores = [x[0] for x in easydict.roc_auc]
        target_scores = [x[1] for x in easydict.roc_auc]
        roc_auc = roc_auc_score(target_scores, float_scores)
        print(
            "ROC-AUC-score on mixed examples (%): {:.3f}".format(
                roc_auc * 100.0
            )
        )
    
    
    if has_classifications:
        for classification in int_labels:
            classification_str = f"_{str(classification)}"
            number_string = f"nb_test{classification_str}"
            correct_string = f"correct{classification_str}"
            fp_string = f"false_positives{classification_str}"
            fn_string = f"false_negatives{classification_str}"
            tp_string = f"true_positives{classification_str}"
            tn_string = f"true_negatives{classification_str}"
            roc_auc_string = f"roc_auc{classification_str}"
            
            print(f"\tPrinting results for {classification}\n")
            print("\t-------------------------------")
            print(
                "\ttest acc on mixed examples (%): {:.3f}".format(
                    easydict[correct_string] / easydict[number_string] * 100.0
                )
            )
            fpr = easydict[fp_string] / (easydict[fp_string]  + easydict[tn_string] )
            print(
                "\tRate of false positives on mixed examples (%): {:.3f}".format(
                    fpr * 100.0
                )
            )

            fnr = easydict[fn_string] / (easydict[fn_string] + easydict[tp_string])
            print(
                "\tRate of false negatives on mixed examples (%): {:.3f}".format(
                    fnr * 100.0
                )
            )
            ## Add small eps to prevent div by 0 without altering the data much
            precision = easydict[tp_string] / (easydict[tp_string] + easydict[fp_string])
            print(
                "\tRate of precision on mixed examples (%): {:.3f}".format(
                    precision * 100.0
                )
            )
            ## Add small eps to prevent div by 0 without altering the data much
            recall = easydict[tp_string] / (easydict[tp_string] + easydict[fn_string])
            print(
                "\tRate of recall on mixed examples (%): {:.3f}".format(
                    recall * 100.0
                )
            )
            ## Add small eps to prevent div by 0 without altering the data much
            f1_score = 2 * ((precision * recall) / (precision + recall))
            print(
                "\tF-1 Score on mixed examples (%): {:.3f}".format(
                    f1_score * 100.0
                )
            )
            if uses_roc_auc:
                float_scores = [x[0] for x in easydict[roc_auc_string]]
                target_scores = [x[1] for x in easydict[roc_auc_string]]
                roc_auc = roc_auc_score(target_scores, float_scores)
                print(
                    "\tROC-AUC-score on mixed examples (%): {:.3f}".format(
                        roc_auc * 100.0
                    )
                )
            print("\n")
    print("\n\n")
    







def calculate_metrics(dictionary):
    metrics = {}
    for key in ['combined', 'inanimate', 'pedestrian', 'vehicle']:
        if key == 'combined':
            true_positives = dictionary['true_positives']
            false_positives = dictionary['false_positives']
            false_negatives = dictionary['false_negatives']
            true_negatives = dictionary['true_negatives']
            roc_float_scores = [x[0] for x in dictionary["roc_auc"]]
            roc_target_scores = [x[1] for x in dictionary["roc_auc"]]
        else:
            true_positives = dictionary[f'true_positives_{key}']
            false_positives = dictionary[f'false_positives_{key}']
            false_negatives = dictionary[f'false_negatives_{key}']
            true_negatives = dictionary[f'true_negatives_{key}']
            roc_float_scores = [x[0] for x in dictionary[f"roc_auc_{key}"]]
            roc_target_scores = [x[1] for x in dictionary[f"roc_auc_{key}"]]

        roc_auc = roc_auc_score(roc_target_scores, roc_float_scores)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        fpr = false_positives / (false_positives + true_negatives)
        fnr = false_negatives / (true_positives + false_negatives)

        metrics[key] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fpr': fpr,
            'fnr': fnr,
            'roc_auc':roc_auc,
        }
    return metrics


def normalize_ffts(batch_tensor, eps=1e-8):
    # Normalize the tensor after applying the Fourier transform
    

    batch_min = batch_tensor.min(dim=1, keepdim=True)[0]
    batch_max = batch_tensor.max(dim=1, keepdim=True)[0]
    batch_tensor = (batch_tensor - batch_min) / (batch_max - batch_min + eps)
    return batch_tensor

def load_rust_threshold_lib(rust_target_file):


    rustlib = ctypes.cdll.LoadLibrary(rust_target_file)

    find_optimal_threshold_combination_ffi = rustlib.find_optimal_threshold_combination_ffi
    find_optimal_threshold_combination_ffi.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    find_optimal_threshold_combination_ffi.restype = None

    return find_optimal_threshold_combination_ffi

def calculate_optimal_threshold_with_rust(preds_list_1, preds_list_2, rust_func, num_thresholds=100, lower_bound_1 = 0.0, lower_bound_2 = 0.0, upper_bound_1=1.0, upper_bound_2=1.0):
    # Call the Rust function from Python
    preds_1_probs = np.array([x[0] for x in preds_list_1], dtype=np.float32)
    preds_1_labels = np.array([x[1] for x in preds_list_1], dtype=np.uint8)
    preds_2_probs = np.array([x[0] for x in preds_list_2], dtype=np.float32)
    preds_2_labels = np.array([x[1] for x in preds_list_2], dtype=np.uint8)
    result = np.empty(2, dtype=np.float32)
    rust_func(
        preds_1_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        preds_1_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        len(preds_1_probs),
        preds_2_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        preds_2_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        len(preds_2_probs),
        num_thresholds,
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(lower_bound_1),
        ctypes.c_float(lower_bound_2),
        ctypes.c_float(upper_bound_1),
        ctypes.c_float(upper_bound_2),
        
    )
    print(f"Optimal threshold for spectral detector in recovery block: {result[0]}")
    print(f"Optimal threshold for binary detector in recovery block: {result[1]}")
    return result[0], result[1]


## This generates 500 thresholds and calculates the F1 score for each of them. It then returns the threshold with the best performance.
def calculate_best_standalone_threshold(preds, stdout_name):
    print(f"Calculating optimal threshold for {stdout_name}")
    thresholds = np.arange(0.01, 1, 0.002)
    f1_scores = [f1_score([x[1] for x in preds], [x[0] for x in preds] >= t) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    return optimal_threshold


def calculate_pred_list_metrics(predictions, threshold):
    metrics = {}
    attack_types = ['APGD', 'Carlini', 'FGSM', 'Shadow', 'Benign']
    
    def init_attack_dict():
        return {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0, 'true_negatives': 0}

    dictionary = {attack_type: init_attack_dict() for attack_type in attack_types}
    
    for pred_float, ground_truth, _, attack_type in predictions:
        
        pred_label = int(pred_float >= threshold)
        attack_type_str = attack_types[attack_type]
        
        if pred_label == ground_truth:
            if pred_label == 1:
                dictionary[attack_type_str]['true_positives'] += 1
            else:
                dictionary[attack_type_str]['true_negatives'] += 1
        else:
            if pred_label == 1:
                dictionary[attack_type_str]['false_positives'] += 1
            else:
                dictionary[attack_type_str]['false_negatives'] += 1
        


    for key in attack_types:
        data = dictionary[key]
        ## need to flip this, as true negatives are now true positives when dealing with benign images.
        if key == "Benign":
            true_positives = data['true_negatives']
            true_negatives = data['true_positives']
        else:
            true_positives = data['true_positives']
            true_negatives = data['true_negatives']
        false_positives = data['false_positives']
        false_negatives = data['false_negatives']
        

        epsilon = 1e-12
        precision = true_positives / (true_positives + false_positives + epsilon)
        recall = true_positives / (true_positives + false_negatives + epsilon)
        f1_score = 2 * ((precision * recall) / (precision + recall + epsilon))
        fpr = false_positives / (false_positives + true_negatives + epsilon)
        fnr = false_negatives / (true_positives + false_negatives + epsilon)

        metrics[key] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fpr': fpr,
            'fnr': fnr,
            }

    return metrics

def calculate_pred_list_metrics_recovery(predictions, threshold, threshold_2=None):
    metrics = {}
    attack_types = ['APGD', 'Carlini', 'FGSM', 'Shadow', 'Benign']
    
    def init_attack_dict():
        return {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0, 'true_negatives': 0}

    dictionary = {attack_type: init_attack_dict() for attack_type in attack_types}
    
    if threshold_2 is not None:
        for pred_float, ground_truth, _, attack_type,passed_initial_detector in predictions:
            if passed_initial_detector == 1:
                pred_label = int(pred_float >= threshold_2)
            else:
                pred_label = int(pred_float >= threshold)
            attack_type_str = attack_types[attack_type]
            
            if pred_label == ground_truth:
                if pred_label == 1:
                    dictionary[attack_type_str]['true_positives'] += 1
                else:
                    dictionary[attack_type_str]['true_negatives'] += 1
            else:
                if pred_label == 1:
                    dictionary[attack_type_str]['false_positives'] += 1
                else:
                    dictionary[attack_type_str]['false_negatives'] += 1
        
    else:
        for pred_float, ground_truth, _, attack_type in predictions:
            
            pred_label = int(pred_float >= threshold)
            attack_type_str = attack_types[attack_type]
            
            if pred_label == ground_truth:
                if pred_label == 1:
                    dictionary[attack_type_str]['true_positives'] += 1
                else:
                    dictionary[attack_type_str]['true_negatives'] += 1
            else:
                if pred_label == 1:
                    dictionary[attack_type_str]['false_positives'] += 1
                else:
                    dictionary[attack_type_str]['false_negatives'] += 1
        


    for key in attack_types:
        data = dictionary[key]
        ## need to flip this, as true negatives are now true positives when dealing with benign images.
        if key == "Benign":
            true_positives = data['true_negatives']
            true_negatives = data['true_positives']
        else:
            true_positives = data['true_positives']
            true_negatives = data['true_negatives']
        false_positives = data['false_positives']
        false_negatives = data['false_negatives']
        

        epsilon = 1e-12
        precision = true_positives / (true_positives + false_positives + epsilon)
        recall = true_positives / (true_positives + false_negatives + epsilon)
        f1_score = 2 * ((precision * recall) / (precision + recall + epsilon))
        fpr = false_positives / (false_positives + true_negatives + epsilon)
        fnr = false_negatives / (true_positives + false_negatives + epsilon)

        metrics[key] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fpr': fpr,
            'fnr': fnr,
            }

    return metrics

