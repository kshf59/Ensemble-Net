import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss
import torch.nn as nn



@torch.inference_mode()
def evaluate(net, dataloader, _criterion_, device, model_name, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    unet_dice_score = 0
    segnet_dice_score = 0
    enet_dice_score = 0
    voting_dice_score = 0
    
    #_criterion_ = nn.CrossEntropyLoss()
    
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        #for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        for batch in dataloader:
            image, mask_true = batch['image'], batch['mask']
            
                    
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            
                
            if model_name == 'ensemble_voting':
                upred, spred, epred = net(image)
                #dpred = dpred['out']
            else:
                mask_pred = net(image)
                if isinstance(mask_pred, OrderedDict):
                    mask_pred = mask_pred['out']
                
            try:
                mn_clss = net.n_classes
            except:
                mn_clss = net.classifier[-1].out_channels
                

            if model_name == 'ensemble_voting':
                assert mask_true.min() >= 0 and mask_true.max()
                
                mask_true2 = F.one_hot(mask_true, mn_clss).permute(0, 3, 1, 2).float()
                upred2 = F.one_hot(upred.argmax(dim=1), mn_clss).permute(0, 3, 1, 2).float()
                spred2 = F.one_hot(spred.argmax(dim=1), mn_clss).permute(0, 3, 1, 2).float()
                epred2 = F.one_hot(epred.argmax(dim=1), mn_clss).permute(0, 3, 1, 2).float()
                  
                vot = (F.softmax(upred2, dim=1) + F.softmax(spred2, dim=1) + F.softmax(epred2, dim=1)) / 3.0
                vot2 = F.one_hot(vot.argmax(dim=1), mn_clss).permute(0, 3, 1, 2).float()
                

                unet_dice_score += multiclass_dice_coeff(upred2[:, 1:], mask_true2[:, 1:], reduce_batch_first=False)
                segnet_dice_score += multiclass_dice_coeff(spred2[:, 1:], mask_true2[:, 1:], reduce_batch_first=False)
                enet_dice_score += multiclass_dice_coeff(epred2[:, 1:], mask_true2[:, 1:], reduce_batch_first=False)
                
                voting_dice_score += multiclass_dice_coeff(vot2[:, 1:], mask_true2[:, 1:], reduce_batch_first=False)
                
                

            else:
                mask_true2 = F.one_hot(mask_true, mn_clss).permute(0, 3, 1, 2).float()
                mask_pred2 = F.one_hot(mask_pred.argmax(dim=1), mn_clss).permute(0, 3, 1, 2).float()

                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred2[:, 1:], mask_true2[:, 1:], reduce_batch_first=False)
                
        if model_name == 'ensemble_voting':
            
            with torch.no_grad():
                valloss = _criterion_(vot, mask_true)
                valloss += dice_loss(
                    vot,
                    F.one_hot(mask_true, mn_clss).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
                losses = valloss.cpu().detach().numpy()            
            
        else:
            
            with torch.no_grad():
                valloss = _criterion_(mask_pred, mask_true)
                valloss += dice_loss(
                    F.softmax(mask_pred, dim=1).float(),
                    F.one_hot(mask_true, mn_clss).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
                losses = valloss.cpu().detach().numpy()
                  
    net.train()
    
    if model_name == 'ensemble_voting':
        
        uresult = unet_dice_score / max(num_val_batches, 1)
        sresult = segnet_dice_score / max(num_val_batches, 1)
        eresult = enet_dice_score / max(num_val_batches, 1)
        votresult = voting_dice_score / max(num_val_batches, 1)
        
        return uresult, sresult, eresult, votresult, losses
    
    else:
        dice_result = dice_score / max(num_val_batches, 1)
        
        return dice_result, losses
    
    
    