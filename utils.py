import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from tabulate import tabulate

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc
    
def cls_acc_per_class(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    # Initialize a dictionary to store the accuracy per class
    class_acc = {}
    
    # Get the list of unique classes in the target
    unique_classes = target.unique()
    
    for cls in unique_classes:
        # Mask for the current class
        class_mask = target == cls
        # Number of samples in the current class
        class_count = class_mask.sum().item()
        
        if class_count == 0:
            # If there are no samples for this class, skip it
            continue
        
        # Correct predictions for the current class
        correct_class = correct[:, class_mask].float().sum().item()
        
        # Calculate accuracy for the current class
        acc = 100 * correct_class / class_count
        
        # Store the accuracy in the dictionary
        class_acc[cls.item()] = acc
    
    return class_acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings *= clip_model.logit_scale.exp()
            clip_weights.append(class_embeddings)
            
            
        text_feature = torch.stack(clip_weights, dim=1).cuda()
        #shape of prompt_num*class_num*d
    return text_feature


def get_clip_logits(images, clip_model, text_feature,topk):
    with torch.no_grad():
        data_type= text_feature.dtype

        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()       
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    
        
        logits_matrix = torch.einsum('pcd,nd -> pcn',text_feature,image_features)
        
        return logits_matrix

def get_res_logits(images ,clip_model,text_feature, weights):
    with torch.no_grad():
        data_type= clip_model.dtype
        weights = weights.type(data_type)
        weights = weights.cuda()
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()       
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        #weighted text feautures
        #weights = torch.full((247,10), 1, dtype=data_type).cuda()
        text_features = torch.einsum('pc,pcd -> cd',weights,text_feature)

        logits =  image_features @ text_features.t()


        return logits
def get_res_logits_zpe(images ,clip_model,text_feature, weights):
    with torch.no_grad():
        data_type= clip_model.dtype
        weights = weights.type(data_type)
        weights = weights.cuda()
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()       
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        #weighted text feautures
        #weights = torch.full((247,10), 1, dtype=data_type).cuda()
       
        text_features = torch.einsum('p,pcd -> cd',weights,text_feature)

        logits =  image_features @ text_features.t()


        return logits

def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess

def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        #dataset = ImageNet(root_path, preprocess)
        dataset = build_dataset(f"imagenet", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=512, is_train=False, tfm=preprocess, shuffle=True)
    
    elif dataset_name in ['A','V','R','S']:
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=512, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=512, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    #print the information of this dataset
    table = []
    table.append(["Dataset", dataset_name])   
    table.append(["Classes", f"{dataset.num_classes}"])
    table.append(["Test Size", f"{len(dataset.test)}"])
    
    print(tabulate(table))

    return test_loader, dataset.classnames, dataset.template