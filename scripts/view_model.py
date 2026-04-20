from pathlib import Path

from torchinfo import summary

from src.config import feature_config as f
from src.config import model_config as m
from src.config import experiment_config as e
from src.models.model import build_embedding_model


def count_params(model) -> tuple[int, int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def main() -> None:
    model = build_embedding_model(
        m.MODEL_NAME,
        n_feats=f.N_MELS,
        emb_dim=m.EMB_DIM,
        dropout=m.DROPOUT,
        conformer_d_model=m.CONFORMER_D_MODEL,
        conformer_dropout=m.CONFORMER_DROPOUT,
        conformer_num_heads=m.CONFORMER_NUM_HEADS,
        conformer_ff_mult=m.CONFORMER_FF_MULT,
        conformer_conv_kernel_size=m.CONFORMER_CONV_KERNEL_SIZE,
        conformer_num_blocks=m.CONFORMER_NUM_BLOCKS,
        ecapa_channels=m.ECAPA_CHANNELS,
        ecapa_mfa_channels=m.ECAPA_MFA_CHANNELS,
        ecapa_attention_channels=m.ECAPA_ATTENTION_CHANNELS,
        ecapa_scale=m.ECAPA_SCALE,
        ecapa_se_bottleneck=m.ECAPA_SE_BOTTLENECK,
        ecapa_dropout=m.ECAPA_DROPOUT,
    )
    model.eval()

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
