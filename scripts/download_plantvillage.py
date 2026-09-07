from datasets import load_dataset
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path(r"C:\Datasets\PlantVillage")


def main():

    print("=" * 70)
    print("🌱 PLANTVILLAGE DATASET DOWNLOADER")
    print("=" * 70)

    print("\n📥 Loading PlantVillage from Hugging Face...")

    print(
        "\nThe current dataset uses the default configuration."
    )

    dataset = load_dataset(
        "mohanty/PlantVillage"
    )

    print("\n✅ Dataset downloaded successfully!")

    print(dataset)

    print("\n📊 Dataset information:")

    for split_name in dataset:

        print(
            f"{split_name}: "
            f"{len(dataset[split_name])} images"
        )

    # --------------------------------------------------------
    # Save dataset locally
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_path = OUTPUT_DIR / "hf_dataset"

    print(
        f"\n💾 Saving dataset to:\n{save_path}"
    )

    dataset.save_to_disk(
        str(save_path)
    )

    print("\n" + "=" * 70)

    print("✅ DOWNLOAD COMPLETE")

    print("=" * 70)

    print(
        f"\nDataset location:\n{save_path}"
    )


if __name__ == "__main__":
    main()