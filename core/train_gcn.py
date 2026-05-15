
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score


def train_and_evaluate_gcn(args, dataset, data, Net):

    model = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = model.to(device), data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f}")

    # final evaluation(full graph)
    model.eval()
    logits = model(data)
    preds = logits.argmax(dim=1).cpu().numpy()
    probs = torch.exp(logits).detach().cpu().numpy()
    true = data.true_y.cpu().numpy()

    accuracy = (preds == true).mean()
    f1 = f1_score(true, preds, average="macro")
    roc_auc = None
    if dataset.num_classes == 2:
        roc_auc = roc_auc_score(true, probs[:, 1])
    cm = confusion_matrix(true, preds)

    print("\n=== Final GCN Evaluation ===")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Macro-F1      : {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC       : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return accuracy, f1, roc_auc, cm, model
