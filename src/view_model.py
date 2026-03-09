from pathlib import Path

import constants as c
from model import CNN1DNET
from torchinfo import summary


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def main():
    num_classes = 200  # current number of train speakers

    model = CNN1DNET(
        n_feats=c.N_MELS,
        num_classes=num_classes,
        emb_dim=c.EMB_DIM,
        dropout=0.3,
    )

    total, trainable, non_trainable = count_params(model)

    s = summary(
        model,
        input_size=(1, c.MAX_FRAMES, c.N_MELS),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=4,
        verbose=0,
    )

    print("MODEL ARCHITECTURE\n")
    print(model)
    print("\nMODEL SUMMARY\n")
    print(s)

    print("\nPARAMETER COUNTS")
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")

    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "model_architecture.txt", "w", encoding="utf-8") as f:
        f.write("MODEL ARCHITECTURE\n\n")
        f.write(str(model))
        f.write("\n\nMODEL SUMMARY\n\n")
        f.write(str(s))
        f.write("\n\nPARAMETER COUNTS\n")
        f.write(f"Total params: {total:,}\n")
        f.write(f"Trainable params: {trainable:,}\n")
        f.write(f"Non-trainable params: {non_trainable:,}\n")

    print(f"\nSaved to: {out_dir / 'model_architecture.txt'}")


if __name__ == "__main__":
    main()
