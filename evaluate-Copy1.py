import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, model_name, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    unet_dice_score = 0
    segnet_dice_score = 0
    enet_dice_score = 0
    voting_dice_score = 0

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
                
                
            '''    
            if mn_clss == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < mn_clss, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, mn_clss).permute(0, 3, 1, 2).float()
            '''
                
            if model_name == 'ensemble_voting':
                assert mask_true.min() >= 0 and mask_true.max()
                
                mask_true = F.one_hot(mask_true, mn_clss).permute(0, 3, 1, 2).float()
                upred = F.one_hot(upred.argmax(dim=1), mn_clss).permute(0, 3, 1, 2).float()
                spred = F.one_hot(spred.argmax(dim=1), mn_clss).permute(0, 3, 1, 2).float()
                epred = F.one_hot(epred.argmax(dim=1), mn_clss).permute(0, 3, 1, 2).float()
                
                vot = (F.softmax(upred, dim=1) + F.softmax(spred, dim=1) + F.softmax(epred, dim=1)) / 3.0
                vot = F.one_hot(vot.argmax(dim=1), mn_clss).permute(0, 3, 1, 2).float()
                

                unet_dice_score += multiclass_dice_coeff(upred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                segnet_dice_score += multiclass_dice_coeff(spred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                enet_dice_score += multiclass_dice_coeff(epred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                
                voting_dice_score += multiclass_dice_coeff(vot[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                
                

            else:
                mask_true = F.one_hot(mask_true, mn_clss).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), mn_clss).permute(0, 3, 1, 2).float()

                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    
    if model_name == 'ensemble_voting':
        
        uresult = unet_dice_score / max(num_val_batches, 1)
        sresult = segnet_dice_score / max(num_val_batches, 1)
        eresult = enet_dice_score / max(num_val_batches, 1)
        votresult = voting_dice_score / max(num_val_batches, 1)
        
        return uresult, sresult, eresult, votresult
    
    else:
        dice_result = dice_score / max(num_val_batches, 1)
        
        return dice_result
    
    
    