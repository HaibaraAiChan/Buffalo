import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.nn import CuGraphSAGEConv
from ogb.nodeproppred import DglNodePropPredDataset
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../../micro_batch_train')
sys.path.insert(0,'../../../models')
sys.path.insert(0,'../../../utils')
from memory_usage import see_memory_usage, nvidia_smi_usage

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(CuGraphSAGEConv(in_size, hid_size, "mean"))
        self.layers.append(CuGraphSAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(CuGraphSAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


# def evaluate(model, graph, dataloader):
#     model.eval()
#     ys = []
#     y_hats = []
#     for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
#         with torch.no_grad():
#             x = blocks[0].srcdata["feat"]
#             ys.append(blocks[-1].dstdata["label"])
#             y_hats.append(model(blocks, x))
#     num_classes = y_hats[0].shape[1]
#     return MF.accuracy(
#         torch.cat(y_hats),
#         torch.cat(ys),
#         task="multiclass",
#         num_classes=num_classes,
#     )


# def layerwise_infer(device, graph, nid, model, batch_size):
#     model.eval()
#     with torch.no_grad():
#         pred = model.inference(
#             graph, device, batch_size
#         )  # pred in buffer_device
#         pred = pred[nid]
#         label = graph.ndata["label"][nid].to(pred.device)
#         num_classes = pred.shape[1]
#         return MF.accuracy(
#             pred, label, task="multiclass", num_classes=num_classes
#         )


def train(args, device, g, dataset, model):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        [10, 25, 30],  # fanout for [layer-0, layer-1, layer-2]
        # prefetch_node_feats=["feat"],
        # prefetch_labels=["label"],
    )
    from cugraph_dgl.convert import cugraph_storage_from_heterograph
    see_memory_usage("----------------------------------------before graph to cugraph ")
    cugraph_g = cugraph_storage_from_heterograph(g)
    see_memory_usage("----------------------------------------after graph to cugraph")
    full_batch_size = len(train_idx)
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        cugraph_g,
        train_idx,
        sampler,
        device=device,
        batch_size=full_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    # val_dataloader = DataLoader(
    #     g,
    #     val_idx,
    #     sampler,
    #     device=device,
    #     batch_size=1024,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=0,
    #     use_uva=use_uva,
    # )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        see_memory_usage("----------------------------------------before train dataloader ")
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            see_memory_usage("----------------------------------------after train dataloader ")
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            see_memory_usage("----------------------------------------before model ")
            y_hat = model(blocks, x)
            see_memory_usage("----------------------------------------after model ")
            loss = F.cross_entropy(y_hat, y)
            
            opt.zero_grad()
            loss.backward()
            see_memory_usage("----------------------------------------after backward ")
            opt.step()
            see_memory_usage("----------------------------------------after optimizer.step ")
            total_loss += loss.item()
            print()
        print("Epoch {:05d} | Loss {:4f} ".format(epoch, total_loss))
        # acc = evaluate(model, g, val_dataloader)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
        #         epoch, total_loss / (it + 1), acc.item()
        #     )
        # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="puregpu",
        choices=["mixed", "puregpu"],
        help="Training mode. 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset(name=args.dataset, root="/home/cc/dataset"))
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 128, out_size).to(device)

    # model training
    print("Training...")
    train(args, device, g, dataset, model)

    # # test the model
    # print("Testing...")
    # acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    # print("Test Accuracy {:.4f}".format(acc.item()))