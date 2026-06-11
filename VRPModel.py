import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = VRP_Encoder(**model_params)
        self.decoder = VRP_Decoder(**model_params)
        self.encoded_nodes = None

        self.use_icl: bool = bool(self.model_params.get('use_icl', False))
        self.prev_node: Optional[torch.Tensor] = None

    def set_edge_context(self, edge_pos, edge_neg,
                         trigram_pos=None, trigram_neg=None,
                         cooccur_pos=None, cooccur_neg=None):
        self.decoder.set_edge_context(
            edge_pos, edge_neg,
            trigram_pos, trigram_neg,
            cooccur_pos, cooccur_neg)

    def clear_context(self):
        self.decoder.clear_context()

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_earlyTW = reset_state.node_earlyTW
        node_lateTW = reset_state.node_lateTW
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        node_TW = torch.cat((node_earlyTW[:, :, None], node_lateTW[:, :, None]), dim=2)
        node_xy_demand_TW = torch.cat((node_xy_demand, node_TW), dim=2)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_TW)
        self.decoder.set_kv(self.encoded_nodes)

        # ★ v7: 重置prev_node追踪
        self.prev_node = None

    def forward(self, state, temperature=1.0):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))
            # prev_node stays None

        elif state.selected_count == 1:
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            # ★ v7: 记录当前节点(depot)作为下一步的prev_node
            self.prev_node = state.current_node.clone()

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)

            # ★ v7: 传递prev_node给decoder
            prev_node = self.prev_node
            self.prev_node = state.current_node.clone()

            probs = self.decoder(encoded_last_node, state.load, state.time,
                                 state.length, state.route_open,
                                 ninf_mask=state.ninf_mask,
                                 current_node=state.current_node,
                                 prev_node=prev_node,
                                 temperature=temperature)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes




class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_TW):
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand_TW)
        out = torch.cat((embedded_depot, embedded_node), dim=1)
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3




class AdjudicatingGate(nn.Module):

    def __init__(self, state_dim=4, edge_feat_dim=9, hidden_dim=32, bias_init=-1.5):
        super().__init__()
        input_dim = state_dim + edge_feat_dim + 3  # +3 是方案 B 新增
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # 保守初始化: 第一层适中 gain, 最后一层小 gain + 明显的负 bias
        nn.init.xavier_uniform_(self.net[0].weight, gain=0.5)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[3].weight, gain=0.1)
        nn.init.constant_(self.net[3].bias, bias_init)

    def forward(self, state_feats, edge_features, mh_norm, ctx_norm, disagree):
        
        gate_input = torch.cat([
            state_feats,
            edge_features,
            mh_norm.unsqueeze(-1),
            ctx_norm.unsqueeze(-1),
            disagree.unsqueeze(-1),
        ], dim=-1)
        return torch.sigmoid(self.net(gate_input)).squeeze(-1)




class VRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # 原始路由策略 (未修改)
        self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None
        self.v = None
        self.single_head_key = None

       
        self.use_icl: bool = bool(self.model_params.get('use_icl', False))
        self.contrast_weight: float = float(self.model_params.get('contrast_weight', 1.0))
        self.use_trigram: bool = bool(self.model_params.get('use_trigram', True))
        self.use_cooccur: bool = bool(self.model_params.get('use_cooccur', True))

        # 矩阵存储
        self.edge_pos: Optional[torch.Tensor] = None
        self.edge_neg: Optional[torch.Tensor] = None
        self.trigram_pos: Optional[torch.Tensor] = None
        self.trigram_neg: Optional[torch.Tensor] = None
        self.cooccur_pos: Optional[torch.Tensor] = None
        self.cooccur_neg: Optional[torch.Tensor] = None

        if self.use_icl:
            # 动态计算特征维度
            edge_feat_dim = 3  # bigram: pos, neg, contrast
            if self.use_trigram:
                edge_feat_dim += 3  # trigram: pos, neg, contrast
            if self.use_cooccur:
                edge_feat_dim += 3  # cooccur: pos, neg, contrast
            self._edge_feat_dim = edge_feat_dim

            # Edge Context Encoder (v6的2a, 扩展到高阶特征)
            edge_hidden_dim = int(self.model_params.get('edge_encoder_hidden', 64))
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feat_dim, edge_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_hidden_dim, embedding_dim),
            )
            nn.init.xavier_uniform_(self.edge_encoder[0].weight, gain=0.5)
            nn.init.zeros_(self.edge_encoder[0].bias)
            nn.init.xavier_uniform_(self.edge_encoder[2].weight, gain=0.1)
            nn.init.zeros_(self.edge_encoder[2].bias)

            # Context score缩放
            ctx_scale_init = float(self.model_params.get('ctx_scale_init', 1.0))
            self.ctx_scale = nn.Parameter(torch.tensor(ctx_scale_init))

            
            gate_hidden = int(self.model_params.get('edge_gate_hidden', 32))
            gate_bias = float(self.model_params.get('edge_gate_bias_init', -1.5))
            self.edge_gate = AdjudicatingGate(
                state_dim=4,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=gate_hidden,
                bias_init=gate_bias,
            )

        # ★ Gate recording for analysis
        self._collecting_gates = False
        self._gate_history = []  # list of (B, P, N) tensors
        # ★ Gate override (for MGC evaluation)
        self.gate_override = None

        
        self._collecting_scores = False
        self._mh_score_history = []
        self._ctx_score_history = []

        
        self._collecting_edge_feats = False
        self._edge_feat_history = []

    def set_edge_context(self, edge_pos, edge_neg,
                         trigram_pos=None, trigram_neg=None,
                         cooccur_pos=None, cooccur_neg=None):
        self.edge_pos = edge_pos
        self.edge_neg = edge_neg
        self.trigram_pos = trigram_pos
        self.trigram_neg = trigram_neg
        self.cooccur_pos = cooccur_pos
        self.cooccur_neg = cooccur_neg

    def clear_context(self):
        self.edge_pos = None
        self.edge_neg = None
        self.trigram_pos = None
        self.trigram_neg = None
        self.cooccur_pos = None
        self.cooccur_neg = None

    def start_gate_recording(self):
        self._collecting_gates = True
        self._gate_history = []

    def stop_gate_recording(self):
        self._collecting_gates = False
        history = self._gate_history
        self._gate_history = []
        return history  # list of (B, P, N) tensors

    # ------------ Gate Override (for MGC) ------------
    def set_gate_override(self, value):
        if value is not None:
            assert 0.0 <= float(value) <= 1.0, f"gate_override must be in [0,1], got {value}"
            self.gate_override = float(value)
        else:
            self.gate_override = None

    def clear_gate_override(self):
        self.gate_override = None

    # ------------ Score Recording ------------
    def start_score_recording(self):
        self._collecting_scores = True
        self._mh_score_history = []
        self._ctx_score_history = []

    def stop_score_recording(self):
        self._collecting_scores = False
        mh, ctx = self._mh_score_history, self._ctx_score_history
        self._mh_score_history = []
        self._ctx_score_history = []
        return mh, ctx

    # ------------ Edge Feat Recording ------------
    def start_edge_feat_recording(self):
        self._collecting_edge_feats = True
        self._edge_feat_history = []

    def stop_edge_feat_recording(self):
        self._collecting_edge_feats = False
        h = self._edge_feat_history
        self._edge_feat_history = []
        return h

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last_node, load, time, length, route_open,
                ninf_mask, current_node=None, prev_node=None, temperature=1.0):
        
        head_num = self.model_params['head_num']

        
        input_cat = torch.cat((
            encoded_last_node,
            load[:, :, None],
            time[:, :, None],
            length[:, :, None],
            route_open[:, :, None],
        ), dim=2)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        out_concat = multi_head_attention(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        
        score = torch.matmul(mh_atten_out, self.single_head_key)

        
        has_edge = (self.use_icl
                    and self.edge_pos is not None
                    and current_node is not None)

        if has_edge:
            B, P = current_node.shape
            N = self.edge_pos.size(1)
            device = current_node.device

            
            ci = current_node[:, :, None, None].expand(B, P, 1, N)

            bi_pos_rows = self.edge_pos[:, None, :, :].expand(B, P, N, N) \
                .gather(dim=2, index=ci).squeeze(2)
            bi_neg_rows = self.edge_neg[:, None, :, :].expand(B, P, N, N) \
                .gather(dim=2, index=ci).squeeze(2)
            bi_contrast = bi_pos_rows - self.contrast_weight * bi_neg_rows

          
            feat_list = [bi_pos_rows, bi_neg_rows, bi_contrast]  # 3维

           
            if self.use_trigram and self.trigram_pos is not None and prev_node is not None:
                # 用advanced indexing: trigram[batch, prev, current, :] → (B*P, N)
                batch_idx = torch.arange(B, device=device)[:, None].expand(B, P).reshape(-1)
                prev_flat = prev_node.reshape(-1).long()
                curr_flat = current_node.reshape(-1).long()

                tri_pos_rows = self.trigram_pos[batch_idx, prev_flat, curr_flat, :].reshape(B, P, N)
                tri_neg_rows = self.trigram_neg[batch_idx, prev_flat, curr_flat, :].reshape(B, P, N)
                tri_contrast = tri_pos_rows - self.contrast_weight * tri_neg_rows

                feat_list.extend([tri_pos_rows, tri_neg_rows, tri_contrast])  # +3维
            elif self.use_trigram:
                # prev_node不可用(早期步骤)或trigram矩阵不存在 → 填零
                zeros = torch.zeros(B, P, N, device=device)
                feat_list.extend([zeros, zeros, zeros])

            
            if self.use_cooccur and self.cooccur_pos is not None:
                co_pos_rows = self.cooccur_pos[:, None, :, :].expand(B, P, N, N) \
                    .gather(dim=2, index=ci).squeeze(2)
                co_neg_rows = self.cooccur_neg[:, None, :, :].expand(B, P, N, N) \
                    .gather(dim=2, index=ci).squeeze(2)
                co_contrast = co_pos_rows - self.contrast_weight * co_neg_rows

                feat_list.extend([co_pos_rows, co_neg_rows, co_contrast])  # +3维
            elif self.use_cooccur:
                zeros = torch.zeros(B, P, N, device=device)
                feat_list.extend([zeros, zeros, zeros])

            
            edge_features = torch.stack(feat_list, dim=-1)  # (B, P, N, edge_feat_dim)

            edge_context = self.edge_encoder(edge_features)  # (B, P, N, emb_dim)

            context_score = (mh_atten_out.unsqueeze(2) * edge_context).sum(dim=-1)
            context_score = self.ctx_scale * context_score

            
            state_feats = torch.stack([load, time, length, route_open], dim=-1)
            state_feats_exp = state_feats.unsqueeze(2).expand(B, P, N, 4)

            
            with torch.no_grad():
                
                mh_mean  = score.mean(dim=-1, keepdim=True)
                mh_std   = score.std(dim=-1, keepdim=True).clamp_min(1e-6)
                mh_norm  = (score - mh_mean) / mh_std                      # (B, P, N)

                ctx_mean = context_score.mean(dim=-1, keepdim=True)
                ctx_std  = context_score.std(dim=-1, keepdim=True).clamp_min(1e-6)
                ctx_norm = (context_score - ctx_mean) / ctx_std            # (B, P, N)

                disagree = (mh_norm - ctx_norm).abs()                      # (B, P, N)

            per_edge_gate = self.edge_gate(
                state_feats_exp, edge_features,
                mh_norm, ctx_norm, disagree,
            )

            # ★ NEW: Gate override (for MGC evaluation)
            if self.gate_override is not None:
                per_edge_gate = torch.full_like(per_edge_gate, self.gate_override)

            
            if self._collecting_gates:
                self._gate_history.append(per_edge_gate)

            
            if self._collecting_edge_feats:
                self._edge_feat_history.append(edge_features.detach())

            if self._collecting_scores:
                self._mh_score_history.append(score.detach())
                self._ctx_score_history.append(context_score.detach())

            score = score + per_edge_gate * context_score

        
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / (sqrt_embedding_dim * temperature)
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        return probs




def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, input1, input2):
        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)
        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)
        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
