import torch
import torch.nn as nn
import torch.nn.functional as F


class LRU(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(LRU, self).__init__()
        self.Wp = nn.Linear(embed_size, embed_size)
        self.Up = nn.Linear(hidden_size, embed_size)
        self.wq = nn.Linear(hidden_size, 2)
        self.uq = nn.Linear(embed_size, 2)
        self.cell = nn.GRUCell(embed_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.Wp.weight)
        nn.init.kaiming_normal_(self.Up.weight)
        nn.init.kaiming_normal_(self.wq.weight)
        nn.init.kaiming_normal_(self.uq.weight)

    def forward(self, pre_hidden, pre_prefer, x):
        """ pre_prefer can be none"""
        prefer_gate = torch.sigmoid(self.Wp(x) + self.Up(pre_prefer))
        leap_logits = F.log_softmax(self.wq(pre_hidden) + self.uq(x * prefer_gate), dim=1)
        binary_gate = F.gumbel_softmax(leap_logits, hard=True)[:, :1]
        next_hidden = self.cell(x, pre_hidden) * binary_gate + pre_hidden * (1 - binary_gate)
        return next_hidden, binary_gate


class HLN(nn.Module):
    def __init__(self, total_items, embed_size, hidden_size, num_prefers):
        super(HLN, self).__init__()
        self.embeddings = nn.Embedding(total_items, embed_size)
        self.lru = LRU(embed_size, hidden_size)
        self.W = nn.Linear(hidden_size, embed_size)
        self.num_prefers = num_prefers
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.embeddings.weight)
        nn.init.kaiming_normal_(self.W.weight)

    def init_zeros(self, input):
        return torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)

    def forward(self, item_seqs):
        x = self.embeddings(item_seqs)

        prefers = [self.init_zeros(x)]
        gates = []

        # Loop preferences
        for i in range(self.num_prefers):
            pre_prefer = torch.stack(prefers).sum(0)
            hidden = self.init_zeros(x)

            binary_gates = []
            for j in range(x.size(1)):
                hidden, binary_gate = self.lru(hidden, pre_prefer, x[:, j, :])
                binary_gates.append(binary_gate)

            prefers.append(hidden)
            gates.append(torch.cat(binary_gates, dim=1))

        agg_prefer = torch.stack(prefers).sum(0)

        # Space tranform
        ideal_item = self.W(agg_prefer)
        scores = ideal_item.mm(self.embeddings.weight.t())

        return F.log_softmax(scores, 1), torch.stack(gates, dim=1)
