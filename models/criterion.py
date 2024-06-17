"""
Modified from DETR https://github.com/facebookresearch/detr
"""
import torch
from torch import nn
from misc import nested_tensor_from_tensor_list, get_world_size, interpolate, is_dist_avail_and_initialized
from .segmentation import dice_loss, sigmoid_focal_loss
from utils import flatten_temporal_batch_dims


class SetCriterion(nn.Module):
    """ This class computes the loss for MTTR.
    The process happens in two steps:
        1) we compute the hungarian assignment between the ground-truth and predicted sequences.
        2) we supervise each pair of matched ground-truth / prediction sequences (mask + reference prediction + conditioned IOU)
    """
    def __init__(self, matcher, weight_dict, eos_coef):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the un-referred category
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        # make sure that only loss functions with non-zero weights are computed:
        losses_to_compute = []
        if weight_dict['loss_dice'] > 0 or weight_dict['loss_sigmoid_focal'] > 0:
            losses_to_compute.append('masks')
        if weight_dict['loss_is_referred'] > 0:
            losses_to_compute.append('is_referred')
        if weight_dict['loss_conditioned_iou'] > 0:
            losses_to_compute.append('conditioned_iou')
        self.losses = losses_to_compute

    def forward(self, outputs, targets):
        long_aux_outputs_list, short_aux_outputs_list = outputs[0].pop('aux_outputs', None), outputs[1].pop('aux_outputs', None)
        # compute the losses for the output of the last decoder layer:
        losses = self.compute_criterion(outputs, targets, losses_to_compute=self.losses)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate decoder layer.
        if long_aux_outputs_list is not None:
            aux_losses_to_compute = self.losses.copy()
            for i, _ in enumerate(long_aux_outputs_list):
                #print(aux_losses_to_compute)
                aux_outputs = [long_aux_outputs_list[i], short_aux_outputs_list[i]]
                losses_dict = self.compute_criterion(aux_outputs, targets, aux_losses_to_compute)
                #print(aux_losses_to_compute)
                losses_dict = {k + f'_{i}': v for k, v in losses_dict.items()}
                losses.update(losses_dict)
        #print("losses:", losses)
        return losses

    def compute_criterion(self, outputs, targets, losses_to_compute):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # T & B dims are flattened so loss functions can be computed per frame (but with same indices per video).
        # also, indices are repeated so so the same indices can be used for frames of the same video.
        T = len(targets)
        # tmp
        long_outputs, short_outputs = outputs
        long_outputs, targets = flatten_temporal_batch_dims(long_outputs, targets)
        for k in short_outputs.keys():
            if isinstance(short_outputs[k], torch.Tensor):
                short_outputs[k] = short_outputs[k].flatten(0, 1)
            else:  # list
                short_outputs[k] = [i for step_t in short_outputs[k] for i in step_t]





        #print("o.shape ",outputs["pred_masks"].shape)    # [B,N,H,W]
        #print("t.shape ",len(targets), " ", targets[0]["masks"].shape) # bs, [Num_instances of sample 0, H, W]
        #print(losses_to_compute) # {'masks', 'is_referred'}
        #print(T) #1
        # repeat the indices list T times so the same indices can be used for each video frame
        indices = T * indices

        # Compute the average number of target masks across all nodes, for normalization purposes
        num_masks = sum(len(t["masks"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=indices[0][0].device)
        # total number of instances masks for the whole batch
        #print(num_masks)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        #print(losses_to_compute)
        for loss in losses_to_compute:
            losses.update(self.get_loss(loss, long_outputs, short_outputs, targets, indices, num_masks=num_masks))
        return losses

    def loss_is_referred(self, long_outputs, short_outputs, targets, indices, **kwargs):
        device = long_outputs['pred_is_referred'].device
        bs = long_outputs['pred_is_referred'].shape[0]

        long_pred_is_referred = long_outputs['pred_is_referred'].log_softmax(dim=-1)  # note that log-softmax is used here
        short_pred_is_referred = short_outputs['pred_is_referred'].log_softmax(dim=-1)
        
        
        target_is_referred = torch.zeros_like(long_pred_is_referred)
        # extract indices of object queries that where matched with text-referred target objects
        query_referred_indices = self._get_query_referred_indices(indices, targets)
        # by default penalize compared to the no-object class (last token)
        target_is_referred[:, :, :] = torch.tensor([0.0, 1.0], device=device)
        if 'is_ref_inst_visible' in targets[0]:  # visibility labels are available per-frame for the referred object:
            is_ref_inst_visible_per_frame = torch.stack([t['is_ref_inst_visible'] for t in targets])
            ref_inst_visible_frame_indices = is_ref_inst_visible_per_frame.nonzero().squeeze()
            # keep only the matched query indices of the frames in which the referred object is visible:
            visible_query_referred_indices = query_referred_indices[ref_inst_visible_frame_indices]
            target_is_referred[ref_inst_visible_frame_indices, visible_query_referred_indices] = torch.tensor([1.0, 0.0], device=device)
        else:  # assume that the referred object is visible in every frame:
            target_is_referred[torch.arange(bs), query_referred_indices] = torch.tensor([1.0, 0.0], device=device)

        loss = -(long_pred_is_referred * target_is_referred).sum(-1) - (short_pred_is_referred * target_is_referred).sum(-1)
        # apply no-object class weights:
        eos_coef = torch.full(loss.shape, self.eos_coef, device=loss.device)
        eos_coef[torch.arange(bs), query_referred_indices] = 1.0
        loss = loss * eos_coef
        bs = len(indices)
        loss = loss.sum() / bs  # sum and normalize the loss by the batch size
        losses = {'loss_is_referred': loss}
        # print('r: ', losses['loss_is_referred'].shape)
        # print(losses['loss_is_referred'])
        return losses

    def loss_masks(self, long_outputs, short_outputs, targets, indices, num_masks, **kwargs):
        assert "pred_masks" in long_outputs
        assert "pred_masks" in short_outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        long_src_masks = long_outputs["pred_masks"]
        long_src_masks = long_src_masks[src_idx]
        short_src_masks = short_outputs["pred_masks"]
        short_src_masks = short_src_masks[src_idx]

        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(long_src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        long_src_masks = interpolate(long_src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        long_src_masks = long_src_masks[:, 0].flatten(1)
        short_src_masks = interpolate(short_src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        short_src_masks = short_src_masks[:, 0].flatten(1)


        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(long_src_masks.shape)

        losses = {
            "loss_sigmoid_focal": sigmoid_focal_loss(long_src_masks, target_masks, num_masks) + sigmoid_focal_loss(short_src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(long_src_masks, target_masks, num_masks) + dice_loss(short_src_masks, target_masks, num_masks),
        }
        # print('d: ', losses['loss_dice'].shape)
        # print(losses['loss_dice'])
        return losses

    def loss_conditioned_iou(self, long_outputs, short_outputs, targets, indices, **kwargs):
        assert "pred_masks" in long_outputs
        assert "pred_masks" in short_outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        long_src_masks = long_outputs["pred_masks"]
        long_src_masks = long_src_masks[src_idx]
        short_src_masks = short_outputs["pred_masks"]
        short_src_masks = short_src_masks[src_idx]

        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(long_src_masks)
        target_masks = target_masks[tgt_idx]

        #print("before____long_src_masks: ", long_src_masks.shape) #[Num_instances, H/4, W/4]
        #print("before____short_src_masks: ", short_src_masks.shape)

        long_src_masks = interpolate(long_src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        long_src_masks = long_src_masks[:, 0].flatten(1).sigmoid()
        short_src_masks = interpolate(short_src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        short_src_masks = short_src_masks[:, 0].flatten(1).sigmoid()

        # print("after____long_src_masks: ", long_src_masks.shape) #[Num_instances, H*W]
        # print("after____short_src_masks: ", short_src_masks.shape)

        
        long_pred_masks = long_src_masks>0.5
        short_pred_masks = short_src_masks>0.5
        long_src_masks = long_src_masks * long_pred_masks
        short_src_masks = short_src_masks * short_pred_masks

        numerator = (long_src_masks * short_src_masks).sum(-1)
        denominator = long_src_masks.sum(-1)
        loss = ((numerator + 1.0) / (denominator + 1.0)).mean(0)
        loss = 1 - loss
        losses = {'loss_conditioned_iou': loss}

        # print('c: ', losses['loss_conditioned_iou'].shape)
        # print(losses['loss_conditioned_iou'])

        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @staticmethod
    def _get_query_referred_indices(indices, targets):
        """
        extract indices of object queries that where matched with text-referred target objects
        """
        query_referred_indices = []
        for (query_idxs, target_idxs), target in zip(indices, targets):
            ref_query_idx = query_idxs[torch.where(target_idxs == target['referred_instance_idx'])[0]]
            query_referred_indices.append(ref_query_idx)
        query_referred_indices = torch.cat(query_referred_indices)
        return query_referred_indices

    def get_loss(self, loss, long_outputs, short_outputs, targets, indices, **kwargs):
        loss_map = {
            'masks': self.loss_masks,
            'is_referred': self.loss_is_referred,
            'conditioned_iou': self.loss_conditioned_iou,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](long_outputs, short_outputs, targets, indices, **kwargs)
