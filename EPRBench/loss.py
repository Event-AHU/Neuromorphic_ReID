from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
import torch
import torch.nn.functional as F

loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())

#  The loss function call (this method will be called at each training iteration)
# def loss_function(descriptors,labels):
def loss_function(descriptors, labels,image_global_raw=None, text_global_raw=None, lambda_contrast: float = 0.1, temperature: float = 0.07):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:
        miner_outputs = miner(descriptors, labels)
        tri_loss = loss_fn(descriptors, labels, miner_outputs)
        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        tri_loss = loss_fn(descriptors, labels)
        batch_acc = 0.0
    
    
    loss_i2t = tri_loss.new_zeros(())
    loss_t2i = tri_loss.new_zeros(())
    loss = tri_loss

    if image_global_raw is not None and text_global_raw is not None:
        # normalize the global descriptors
        img = F.normalize(image_global_raw, p=2, dim=1)
        txt = F.normalize(text_global_raw, p=2, dim=1)
        logits = (img @txt.t()) / temperature
        targets = torch.arange(img.shape[0], device=img.device)
        
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)
        
        loss = tri_loss + lambda_contrast * (loss_i2t + loss_t2i)
    return loss, tri_loss, loss_i2t, loss_t2i
