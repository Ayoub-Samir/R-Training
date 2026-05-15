import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCN_Net(nn.Module):

    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


MODEL_REGISTRY = {
    "gcn": GCN_Net,
    "vanilla_gcn": GCN_Net,
}


def get_model_class(model_type):
    key = str(model_type or "gcn").lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]
