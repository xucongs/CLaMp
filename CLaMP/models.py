
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter



class RBFExpansion(nn.Module):
    def __init__(self, num_rbf=16, cutoff=8.0):
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer('freqs', torch.pi * torch.arange(1.0, num_rbf + 1.0))

    def forward(self, dist):
        d = dist / self.cutoff
        rbf = torch.sin(self.freqs * d) / (d + 1e-8)
        envelope = torch.where(
            dist.squeeze(-1) < self.cutoff,
            1.0 - 6 * d.squeeze(-1) ** 5
                + 15 * d.squeeze(-1) ** 4
                - 10 * d.squeeze(-1) ** 3,
            torch.zeros_like(dist.squeeze(-1)),
        )
        return rbf * envelope.unsqueeze(-1)


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim, num_rbf=16, cutoff=8.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_s = nn.LayerNorm(hidden_dim)
        self.rbf = RBFExpansion(num_rbf, cutoff)

        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf, 3 * hidden_dim),
            nn.SiLU(),
            nn.Linear(3 * hidden_dim, 3 * hidden_dim),
            nn.SiLU(),
            nn.Linear(3 * hidden_dim, 3 * hidden_dim),
        )
        self.coord_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Tanh(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(self, s, v, coord, edge_index, edge_vec):
        row, col = edge_index
        N = s.size(0)
        s_norm = self.norm_s(s)

        dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        rbf  = self.rbf(dist)
        msg_input = torch.cat([s_norm[row], s_norm[col], rbf], dim=-1)
        W = self.message_net(msg_input)
        W_s, W_vg, W_vs = torch.split(W, self.hidden_dim, dim=-1)

        delta_s_msg = W_s * s_norm[col]
        e_norm = edge_vec / (dist + 1e-6)
        v_j = v[col]
        v_j_proj = (v_j * e_norm.unsqueeze(1)).sum(dim=-1)
        delta_v_msg = (
            W_vg.unsqueeze(-1) * v_j +
            W_vs.unsqueeze(-1) * v_j_proj.unsqueeze(-1) * e_norm.unsqueeze(1)
        )
        coord_weight = self.coord_net(delta_s_msg)
        delta_coord_msg = edge_vec * coord_weight

        agg_s = scatter(delta_s_msg, row, dim=0, dim_size=N, reduce='add')
        agg_v = scatter(
            delta_v_msg.reshape(-1, self.hidden_dim * 3),
            row, dim=0, dim_size=N, reduce='add'
        ).reshape(N, self.hidden_dim, 3)
        agg_coord = scatter(delta_coord_msg, row, dim=0, dim_size=N, reduce='add')

        upd_input = torch.cat([s_norm, agg_s], dim=-1)
        U = self.update_net(upd_input)
        U_vg, U_s = torch.split(U, self.hidden_dim, dim=-1)

        s_new = s + U_s
        v_new = v + U_vg.unsqueeze(-1) * agg_v
        coord_new = coord + agg_coord * 0.01
        return s_new, v_new, coord_new



class GraphEncoder(nn.Module):
    def __init__(self, in_node_dim=10, hidden_dim=256, out_dim=512,
                 n_layers=4, num_rbf=16, cutoff=8.0):
        super().__init__()
        self.node_encoder = nn.Linear(in_node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, num_rbf, cutoff) for _ in range(n_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, pos, edge_index, edge_vec, batch, **kwargs):
        s = self.node_encoder(x)
        v = torch.zeros(s.size(0), s.size(1), 3,
                        device=s.device, dtype=s.dtype)
        coord = pos.clone()
        for layer in self.layers:
            s, v, coord = layer(s, v, coord, edge_index, edge_vec)
        graph_emb = global_mean_pool(s, batch)
        return self.output_proj(graph_emb)


def _build_property_head(embed_dim, dropout=0.3, n_props=3):
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, embed_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(embed_dim // 2, n_props),
    )



class GraphOnlyModel(nn.Module):
    def __init__(self, in_node_dim=10, hidden_dim=256, embed_dim=512,
                 n_layers=4, num_rbf=16, cutoff=8.0,
                 dropout=0.3, n_props=3):
        super().__init__()
        self.graph_encoder = GraphEncoder(
            in_node_dim=in_node_dim, hidden_dim=hidden_dim, out_dim=embed_dim,
            n_layers=n_layers, num_rbf=num_rbf, cutoff=cutoff,
        )
        self.property_head = _build_property_head(embed_dim, dropout, n_props)

    def forward(self, batch_graph):
        emb = self._encode(batch_graph)
        return self.property_head(emb)

    def _encode(self, bg):
        return self.graph_encoder(
            x=bg.x, pos=bg.pos, edge_index=bg.edge_index,
            edge_vec=bg.edge_vec, batch=bg.batch,
        )

    def predict_properties(self, bg):
        return self.forward(bg)

    def encode_graph(self, bg):
        return self._encode(bg)



class MultimodalModel(nn.Module):
    def __init__(self, in_node_dim=10, hidden_dim=256, embed_dim=512,
                 n_layers=4, llm_dim=2048, num_rbf=16, cutoff=8.0,
                 dropout=0.3, n_props=3, use_projection_head=True):
        super().__init__()
        self.use_projection_head = use_projection_head

        
        self.graph_encoder = GraphEncoder(
            in_node_dim=in_node_dim, hidden_dim=hidden_dim, out_dim=embed_dim,
            n_layers=n_layers, num_rbf=num_rbf, cutoff=cutoff,
        )
        
        self.property_head = _build_property_head(embed_dim, dropout, n_props)

        
        self.text_projector = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, embed_dim),
        )

        
        if use_projection_head:
            self.graph_projector = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.graph_projector = nn.Identity()

    def _encode(self, bg):
        return self.graph_encoder(
            x=bg.x, pos=bg.pos, edge_index=bg.edge_index,
            edge_vec=bg.edge_vec, batch=bg.batch,
        )

    def forward(self, batch_graph, text_emb):
        graph_emb = self._encode(batch_graph)
        pred_props = self.property_head(graph_emb)

        graph_for_contrast = self.graph_projector(graph_emb)
        text_projected     = self.text_projector(text_emb)

        return (
            F.normalize(graph_for_contrast, p=2, dim=-1),
            F.normalize(text_projected,    p=2, dim=-1),
            pred_props,
        )

    
    def encode_graph(self, batch_graph):
        emb = self._encode(batch_graph)
        return F.normalize(self.graph_projector(emb), p=2, dim=-1)

    def encode_text(self, text_emb):
        return F.normalize(self.text_projector(text_emb), p=2, dim=-1)

    def predict_properties(self, batch_graph):
        return self.property_head(self._encode(batch_graph))



def build_model(model_type: str, cfg):
    if model_type == "graph_only":
        return GraphOnlyModel(
            in_node_dim=cfg.IN_NODE_DIM, hidden_dim=cfg.HIDDEN_DIM,
            embed_dim=cfg.EMBED_DIM, n_layers=cfg.N_LAYERS,
            num_rbf=cfg.NUM_RBF, cutoff=cfg.CUTOFF, dropout=cfg.DROPOUT,
        )
    elif model_type == "multimodal":
        return MultimodalModel(
            in_node_dim=cfg.IN_NODE_DIM, hidden_dim=cfg.HIDDEN_DIM,
            embed_dim=cfg.EMBED_DIM, n_layers=cfg.N_LAYERS,
            llm_dim=cfg.LLM_DIM, num_rbf=cfg.NUM_RBF, cutoff=cfg.CUTOFF,
            dropout=cfg.DROPOUT,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    from config import CFG
    for mt in ["graph_only", "multimodal"]:
        m = build_model(mt, CFG)
        n = sum(p.numel() for p in m.parameters())
        print(f"{mt:<14}: {n/1e6:.2f} M parameters")
