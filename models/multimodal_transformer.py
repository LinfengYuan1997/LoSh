"""
MTTR Multimodal Transformer class.
Modified from DETR https://github.com/facebookresearch/detr
"""
import copy
import os
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat
from transformers import RobertaModel, RobertaTokenizerFast
from models.position_encoding_2d import PositionEmbeddingSine2D

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)


class MultimodalTransformer(nn.Module):
    def __init__(self, num_encoder_layers=3, num_decoder_layers=3,
                 text_encoder_type="roberta-base", freeze_text_encoder=True, **kwargs):
        super().__init__()
        self.d_model = kwargs['d_model']
        encoder_layer = TransformerEncoderLayer(**kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(**kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, norm=nn.LayerNorm(self.d_model),
                                          return_intermediate=True)
        self.pos_encoder_2d = PositionEmbeddingSine2D()
        self._reset_parameters()

        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
        self.text_encoder.pooler = None  # this pooler is never used, this is a hack to avoid DDP problems...
        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.txt_proj = FeatureResizer(
            input_feat_size=self.text_encoder.config.hidden_size,
            output_feat_size=self.d_model,
            dropout=kwargs['dropout'],
        )
        self.tc = kwargs['text_condition']

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, vid_embeds, vid_pad_mask, text_queries, short_text_queries, obj_queries):
        device = vid_embeds.device
        t, b, _, h, w = vid_embeds.shape


        # ===================================for long text queries===================================
        txt_memory, txt_pad_mask = self.forward_text(text_queries, device)
        # add temporal dim to txt memory & padding mask:
        txt_memory = repeat(txt_memory, 's b c -> s (t b) c', t=t)
        txt_pad_mask = repeat(txt_pad_mask, 'b s -> (t b) s', t=t)
        _, len_long_query = txt_pad_mask.shape
        #print("txt_memory.shape: ",txt_memory.shape)
        #print("txt_pad_mask.shape: ", txt_pad_mask.shape)
        # ===================================for long text queries===================================

        # ===================================for short text queries===================================
        short_txt_memory, short_txt_pad_mask = self.forward_text(short_text_queries, device)
        short_txt_memory = repeat(short_txt_memory, 's b c -> s (t b) c', t=t)
        short_txt_pad_mask = repeat(short_txt_pad_mask, 'b s -> (t b) s', t=t)
        #print('short_txt_pad:', short_txt_pad_mask)
        # pad the short memory and padding mask
        len_short_query, t_b, c = short_txt_memory.shape
        diff_long_short = len_long_query - len_short_query
        if diff_long_short > 0:
            diff_txt_memory = torch.zeros([diff_long_short,t_b,c], dtype=short_txt_memory.dtype, device=device)
            diff_pad_mask = torch.ones([t_b,diff_long_short], dtype=short_txt_pad_mask.dtype, device=device)
            short_txt_memory = torch.cat((short_txt_memory, diff_txt_memory), dim=0)
            short_txt_pad_mask = torch.cat((short_txt_pad_mask,diff_pad_mask), dim=1)
        elif diff_long_short == 0:
            # print("same length of long and short")
            # print(text_queries, short_text_queries)
            # print(len_long_query, len_short_query)
            pass
        else:
            print("sth wrong about short query annotations!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(text_queries, short_text_queries)
            print(len_long_query, len_short_query)
            diff_long_short = 0 - diff_long_short
            diff_txt_memory = torch.zeros([diff_long_short, t_b, c], dtype=txt_memory.dtype, device=device)
            diff_pad_mask = torch.ones([t_b, diff_long_short], dtype=txt_pad_mask.dtype, device=device)
            txt_memory = torch.cat((txt_memory, diff_txt_memory), dim=0)
            txt_pad_mask = torch.cat((txt_pad_mask, diff_pad_mask), dim=1)
        # print("txt_memory.shape: ",txt_memory.shape)
        # print("short_txt_memory.shape: ", short_txt_memory.shape)
        # print(diff_txt_memory.shape, diff_txt_memory.dtype)
        # print("txt_pad_mask.shape: ", txt_pad_mask.shape)
        # print("short_txt_pad_mask.shape: ", short_txt_pad_mask.shape)
        # print(diff_pad_mask.shape, diff_pad_mask.dtype)
        # print(diff_pad_mask)
        #print("short_txt_memory.shape: ", short_txt_memory.shape)
        #print("short_txt_pad_mask.shape: ", short_txt_pad_mask.shape)
        # ===================================for short text queries===================================



        # ==========get encoder_src_seq_long and short_encoder_scr_seq, then concatenate them together==========

        vid_embeds = rearrange(vid_embeds, 't b c h w -> (h w) (t b) c')
        # Concat the image & text embeddings on the sequence dimension
        long_encoder_src_seq = torch.cat((vid_embeds, txt_memory), dim=0)
        short_encoder_src_seq = torch.cat((vid_embeds.clone(), short_txt_memory), dim=0)

        #print("encoder_src_seq.shape:", encoder_src_seq.shape) #[H*W+S, T*B, C]
        long_seq_mask = torch.cat((rearrange(vid_pad_mask, 't b h w -> (t b) (h w)'), txt_pad_mask), dim=1)
        short_seq_mask = torch.cat((rearrange(vid_pad_mask.clone(), 't b h w -> (t b) (h w)'), short_txt_pad_mask), dim=1)
        #print("seq_mask.shape:", seq_mask.shape) #[T*B, H*W+S]
        # vid_pos_embed is: [T*B, H, W, d_model]
        vid_pos_embed = self.pos_encoder_2d(rearrange(vid_pad_mask, 't b h w -> (t b) h w'), self.d_model)
        # use zeros in place of pos embeds for the text sequence:
        long_pos_embed = torch.cat((rearrange(vid_pos_embed, 't_b h w c -> (h w) t_b c'), torch.zeros_like(txt_memory)), dim=0)
        short_pos_embed = torch.cat((rearrange(vid_pos_embed, 't_b h w c -> (h w) t_b c'), torch.zeros_like(short_txt_memory)), dim=0)

        # concatenate long short seq,pos and mask at T_B dim
        encoder_src_seq = torch.cat((long_encoder_src_seq, short_encoder_src_seq), dim=1)
        seq_mask = torch.cat((long_seq_mask, short_seq_mask), dim=0)
        pos_embed = torch.cat((long_pos_embed, short_pos_embed), dim=1 )
        #print(encoder_src_seq.shape)
        #print(seq_mask.shape)
        #print(pos_embed.shape)
        # ==========get encoder_src_seq_long and short_encoder_scr_seq, then concatenate them together==========

        """ original MTTR
        vid_embeds = rearrange(vid_embeds, 't b c h w -> (h w) (t b) c')
        # Concat the image & text embeddings on the sequence dimension
        encoder_src_seq = torch.cat((vid_embeds, txt_memory), dim=0)
        seq_mask = torch.cat((rearrange(vid_pad_mask, 't b h w -> (t b) (h w)'), txt_pad_mask), dim=1)
        # vid_pos_embed is: [T*B, H, W, d_model]
        vid_pos_embed = self.pos_encoder_2d(rearrange(vid_pad_mask, 't b h w -> (t b) h w'), self.d_model)
        # use zeros in place of pos embeds for the text sequence:
        pos_embed = torch.cat((rearrange(vid_pos_embed, 't_b h w c -> (h w) t_b c'), torch.zeros_like(txt_memory)), dim=0)
        """

        memory = self.encoder(encoder_src_seq, src_key_padding_mask=seq_mask, pos=pos_embed)  # [H*W+S, T*B, C]


        long_vid_memory = rearrange(memory[:h*w, :t*b, :], '(h w) (t b) c -> t b c h w', h=h, w=w, t=t, b=b)
        long_txt_memory = memory[h*w:, :t*b, :]
        long_txt_memory = rearrange(long_txt_memory, 's t_b c -> t_b s c')

        short_vid_memory = rearrange(memory[:h*w, t*b:, :], '(h w) (t b) c -> t b c h w', h=h, w=w, t=t, b=b)
        short_txt_memory = memory[h*w:, t*b:, :]
        short_txt_memory = rearrange(short_txt_memory, 's t_b c -> t_b s c')



        # ==========remove padded txt_memory to get pure txt memory==========
        long_txt_memory = [t_mem[~pad_mask] for t_mem, pad_mask in zip(long_txt_memory, txt_pad_mask)] # remove padding in original MTTR
        short_txt_memory = [t_mem[~pad_mask] for t_mem, pad_mask in zip(short_txt_memory, short_txt_pad_mask)]
        #txt_memory = [t_mem[~pad_mask] for t_mem, pad_mask in zip(txt_memory, torch.cat((txt_pad_mask,short_txt_pad_mask),dim=1) )]  # remove padding in long_short_concatenation MTTR
        # ==========remove padded txt_memory to get pure txt memory==========


        #print("length of padded long short text queries:")
        #for index in range(len(txt_memory)):
        #  print(txt_memory[index].shape)
        #add T*B dims to query embeds (was: [N, C], where N is the number of object queries):
        obj_queries = repeat(obj_queries, 'n c -> n (t b) c', t=t, b=b)

        if self.tc:
          list_mean_txt_memory = [txt_item.mean(dim=0) for txt_item in txt_memory]
          pooled_unpadded_txt_memory = torch.stack(list_mean_txt_memory,dim=0).contiguous()
          tgt = repeat(pooled_unpadded_txt_memory, 't_b c -> n t_b c', n = obj_queries.shape[0])
        else:
          tgt = torch.zeros_like(obj_queries)  # [N, T*B, C]


        # test new
        tgt = repeat(tgt, 'n t_b c -> n (t t_b) c', t=2)
        obj_queries = repeat(obj_queries, 'n t_b c -> n (t t_b) c', t=2)

        # hs is [L, N, T*B, C] where L is number of layers in the decoder
        hs = self.decoder(tgt, memory, memory_key_padding_mask=seq_mask, pos=pos_embed, query_pos=obj_queries)


        #test new
        long_hs = hs[:,:,:t*b,:]
        short_hs = hs[:,:,t*b:,:]

        long_hs = rearrange(long_hs, 'l n (t b) c -> l t b n c', t=t, b=b)
        short_hs = rearrange(short_hs, 'l n (t b) c -> l t b n c', t=t, b=b)

        return long_hs, long_vid_memory, long_txt_memory, short_hs, short_vid_memory, short_txt_memory


    def forward_text(self, text_queries, device):
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', return_tensors='pt')
        tokenized_queries = tokenized_queries.to(device)
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded_text = self.text_encoder(**tokenized_queries)
        # Transpose memory because pytorch's attention expects sequence first
        txt_memory = rearrange(encoded_text.last_hidden_state, 'b s c -> s b c')
        txt_memory = self.txt_proj(txt_memory)  # change text embeddings dim to model dim
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        txt_pad_mask = tokenized_queries.attention_mask.ne(1).bool()  # [B, S]
        return txt_memory, txt_pad_mask

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
