import random
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of WPE on specific dataset in yaml format.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process")
    parser.add_argument('--data-root', dest='data_root', type=str, default='/data/gpfs/projects/punim2161/datasets', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--topk', dest='topk',type=int,default=1, help='the number of the choosen class based on similarity' )
    parser.add_argument('--temp', dest='temp',type=float,default=3.0, help='the temperature' )
    parser.add_argument('--average', dest='average', action='store_true', help='Whether using the average weight for the prompt pools')
    args = parser.parse_args()
    return args

    



def run_test_wpe(loader, clip_model, text_feature,average,topk,temp):
    with torch.no_grad():
        num_prompt,num_class,_ = text_feature.shape
        
        prompt_weight = torch.full((num_prompt,num_class), 1e-10, dtype=torch.float32)
        prompt_weight_zpe = torch.full((num_prompt,), 1e-10, dtype=torch.float32)
        accuracies = []
        accuracies_zpe = []
        accuracies3 = []
        total_count = torch.zeros((num_class), dtype=torch.long)
        total_target_count = torch.zeros((num_class), dtype=torch.long)
        total_count_matrix = torch.zeros((num_prompt,num_class), dtype=torch.long)
        count_zpe = 0
    if average == False:
        for i, (images, target) in enumerate(loader):
            logits = get_clip_logits(images, clip_model, text_feature,topk)
            #test normalised
            img_mean_ds_logits = logits.mean(0, keepdims=True) 
            normalised = logits - img_mean_ds_logits
            topk_values, topk_indices = torch.topk(normalised, topk, dim=1)
            topk_values_zpe, topk_indices_zpe = torch.topk(normalised, 1, dim=1)

            topk_indices=topk_indices.detach().cpu()
            topk_values=topk_values.detach().cpu()
            
            #statisitc the label bias of gt and predicited pesuod lable
            flattened_matrix = topk_indices.view(-1) #
            count = torch.bincount(flattened_matrix,minlength=num_class)

            flattened_values = topk_values.view(topk_indices.size(0), -1)
            flattened_indices = topk_indices.view(topk_indices.size(0), -1)
            counts_matrix = torch.stack([torch.bincount(slice, minlength=num_class) for slice in flattened_indices])
            
            total_count_matrix = total_count_matrix + counts_matrix

            total_count = total_count + count
            target_count = torch.bincount(target,minlength=num_class)
            total_target_count = total_target_count + target_count

            #update the weigth 
            sum_matrix = torch.zeros((num_prompt,num_class), dtype=clip_model.dtype)
            sum_matrix.scatter_add_(1, flattened_indices, flattened_values)
            sum_matrix = sum_matrix.type(prompt_weight.dtype)
            prompt_weight = prompt_weight + sum_matrix

            #zpe
            topk_indices_zpe=topk_indices_zpe.detach().cpu()
            topk_values_zpe=topk_values_zpe.detach().cpu()
            count_zpe = count_zpe + topk_indices_zpe.shape[2]
            #statisitc the label bias of gt and predicited pesuod lable

            flattened_values_zpe = topk_values_zpe.view(topk_indices.size(0), -1)
            

            prompt_weight_zpe = prompt_weight_zpe + flattened_values_zpe.sum(dim=1)
            
        

        safe_total_count_matrix = torch.where(total_count_matrix == 0, 1, total_count_matrix)
        prompt_weight = prompt_weight / (temp*safe_total_count_matrix)

        prompt_weight_zpe = prompt_weight_zpe / (temp*count_zpe)


    
        
       
        

        

    #test
    for i, (images, target) in enumerate(loader):
        weights =  F.softmax(prompt_weight, dim=0)
        logits = get_res_logits(images ,clip_model,text_feature,weights)
        target = target.cuda()
        acc = cls_acc(logits, target) 
        accuracies.append(acc)

       
        
        zpe_weights = F.softmax(prompt_weight_zpe, dim=0)
        logits = get_res_logits_zpe(images ,clip_model,text_feature,zpe_weights)
        target = target.cuda()
        acc_zpe = cls_acc(logits, target) 
        accuracies_zpe.append(acc_zpe)

        


   
    return sum(accuracies)/len(accuracies), sum(accuracies_zpe)/len(accuracies_zpe)





            

   

        

def calculate_standard_deviation(acc_list):
    n = len(acc_list)
    mean_value = sum(acc_list) / n
    variance = sum((x - mean_value) ** 2 for x in acc_list) / (n - 1)
    std_dev = variance ** 0.5
    return std_dev

def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    #random.seed(args.seed)
    #torch.manual_seed(args.seed)
    
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        acc_wpe_list = []
        acc_avg_list = []
        acc_zpe_list = []
        print(f"Processing {dataset_name} dataset.")        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        #print("---- The number of the prompts: {}. ----\n".format(num_prompt))
        print("---- The topk: {}. ----\n".format(args.topk))
        print("---- The temperature: {}. ----\n".format(args.temp))

        for seed in [1,2,3,4,5]:
            random.seed(seed)
            torch.manual_seed(seed)
            # Run TDA on each dataset
        
            text_feature = clip_classifier(classnames, template, clip_model)
            acc_wpe,acc_zpe = run_test_wpe(test_loader, clip_model, text_feature, args.average,args.topk,args.temp)
            acc_wpe_list.append(acc_wpe)
            acc_zpe_list.append(acc_zpe)

        print("---- CARPRT's test accuracy: {:.2f}. ----\n".format(sum(acc_wpe_list)/len(acc_wpe_list)))   
        print("---- ZPE's test accuracy: {:.2f}. ----\n".format(sum(acc_zpe_list)/len(acc_zpe_list)))   

        wpe_std = calculate_standard_deviation(acc_wpe_list)
        zpe_std = calculate_standard_deviation(acc_zpe_list)

        print("---- CARPRT's test accuracy standard deviation: {:.2f}. ----\n".format(wpe_std))
        print("---- ZPE's test accuracy standard deviation: {:.2f}. ----\n".format(zpe_std))

       
if __name__ == "__main__":
    main()