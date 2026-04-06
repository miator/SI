from pathlib import Path

from torchinfo import summary

from src.config import feature_config as f
from src.config import model_config as m
from src.config import experiment_config as e
from src.models.model import CNN1DNET


def count_params(model: CNN1DNET) -> tuple[int, int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def main() -> None:
    model = CNN1DNET(
        n_feats=f.N_MELS,
        emb_dim=m.EMB_DIM,
        dropout=m.DROPOUT,
    )

    total, trainable, non_trainable = count_params(model)

    model_summary = summary(
        model,
        input_size=(1, f.MAX_FRAMES, f.N_MELS),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=4,
        verbose=0,
    )

    print("MODEL ARCHITECTURE\n")
    print(model)
    print("\nMODEL SUMMARY\n")
    print(model_summary)

    print("\nPARAMETER COUNTS")
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")

    out_dir = Path(e.RUNS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_architecture.txt"

    with out_path.open("w", encoding="utf-8") as file:
        file.write("MODEL ARCHITECTURE\n\n")
        file.write(str(model))
        file.write("\n\nMODEL SUMMARY\n\n")
        file.write(str(model_summary))
        file.write("\n\nPARAMETER COUNTS\n")
        file.write(f"Total params: {total:,}\n")
        file.write(f"Trainable params: {trainable:,}\n")
        file.write(f"Non-trainable params: {non_trainable:,}\n")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
