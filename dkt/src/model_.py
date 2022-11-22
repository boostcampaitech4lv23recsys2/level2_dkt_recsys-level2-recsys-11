import torch
import torch.nn as nn

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )


class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # cate_x embedding
        self.cate_emb_layer = nn.Embedding(args.offset, args.emb_dim, padding_idx=0)
        self.cate_proj_layer = nn.Sequential(
            nn.Linear(args.emb_dim * args.cate_num, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim)
        )

        # cont_x embedding
        self.cont_proj_layer = nn.Sequential(
            nn.Linear(args.cont_num, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim)
        )

        # cate_x + cont_x projection
        self.comb_proj_layer = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
        )

        # lstm args.n_layer 추가해라
        self.lstm_layer = \
            nn.LSTM(args.hidden_dim, args.hidden_dim, 1, batch_first=True)

        # final layer
        self.final_layer = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim), # 원래는 (args.hidden_dim, args.hidden_dim)
            nn.LayerNorm(args.hidden_dim),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        
    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor):
        # cate forward
        cate_emb_x = self.cate_emb_layer(cate_x)
        cate_emb_x = cate_emb_x.view(cate_emb_x.size(0), self.args.max_seq_len, -1)
        cate_proj_x = self.cate_proj_layer(cate_emb_x)

        # cont forawrd
        cont_proj_x = self.cont_proj_layer(cont_x)

        # comb forward
        comb_x = torch.cat([cate_proj_x, cont_proj_x], dim=2)
        comb_proj_x = self.comb_proj_layer(comb_x)
        
        # lstm forward
        hs, _ = self.lstm_layer(comb_proj_x)

        # batch_first = True 이기 때문에.
        hs = hs.contiguous().view(hs.size(0), -1, self.args.hidden_dim)

        # final forward
        out = self.final_layer(hs)
        return out.squeeze()