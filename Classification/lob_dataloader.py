import webdataset as wds
import torch
from pathlib import Path

def get_dataloader(data_pattern, target_type="cls", batch_size=32, shuffle_size=1000, num_workers=4, is_train=False):
    """
    Create a WebDataset dataloader for sharded tar.gz files.
    
    :
        data_pattern: Path pattern for shards (e.g., "/path/to/data/train-{000000..000999}.tar.gz")
        target_type: "cls" or "reg"
        batch_size: Batch size
        shuffle_size: Shuffle buffer size (training only)
        num_workers: Number of workers
        is_train: Whether to enable shuffling
    """
    assert target_type in ["cls", "reg"], "target_type must be 'cls' or 'reg'"
    
    # Verify files exist
    if not any(Path(data_pattern.split("{")[0]).parent.glob(Path(data_pattern).name.split("{")[0] + "*")):
        raise FileNotFoundError(f"No files found matching pattern: {data_pattern}")
    
    # Build pipeline
    dataset = (
        wds.WebDataset(data_pattern)
        .decode()
        .to_tuple(f"input.npy", f"target.{target_type}", "metadata.json")
    )
    
    # Add shuffling for training
    if is_train:
        dataset = dataset.shuffle(shuffle_size)
    
    # Add batching
    dataset = dataset.batched(batch_size, partial=False)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True
    )

# Example usage with sharded files:
if __name__ == "__main__":
    path_to_data = "/Users/danielfisher/repositories/futfut/webdataset"


    # For sharded training data (000001..000100)
    train_loader = get_dataloader(
        f"{path_to_data}/train-{{000001..000052}}.tar.gz",
        target_type="cls",
        is_train=True
    )
    
    # For sharded validation data
    val_loader = get_dataloader(
        f"{path_to_data}/val-{{000001..000011}}.tar.gz",
        target_type="cls",
        is_train=False
    )

    # For sharded validation data
    test_loader = get_dataloader(
        f"{path_to_data}/test-{{000001..000009}}.tar.gz",
        target_type="cls",
        is_train=False
    )
    
    # Test the loader
    for batch in test_loader:
        inputs, targets, metadata = batch
        print(targets)
        print(f"Batch shape - inputs: {inputs.shape}, targets: {targets.shape}")
        print(f"Targets - {targets}")
        print(f"Sample metadata: {metadata[0]}")


NUM_EPOCHS= 50
CHECKPOINT_PATH= "/kaggle/working/"
PRETRAIN_EPOCHS=50
EMB_DIM=64
DEPTH=3
ASB=True
ADAPTIVE_FILTER=True
ICB=True
LOAD_FROM_PRETRAINED=True
MODEL_ID="AUDUSD"
DATAPATH = "/kaggle/input/market-depth-audusd",
BATCH_SIZE = 128,
NUM_CLASSES = 5,
CLASS_NAMES = [1,2,3,4,5],
SEQ_LEN = 600,
NUM_CHANNELS = 42,
TRAIN_LR=1e-3
PRETRAIN_LR=1e-3
MASKING_RATIO=0.4
DROPOUT_RATE=0.5
PATCH_SIZE=64


# load from checkpoint
run_description = f"{os.path.basename(DATAPATH)}_dim{EMB_DIM}_depth{DEPTH}___"
run_description += f"ASB_{ASB}__AF_{ADAPTIVE_FILTER}__ICB_{ICB}__preTr_{LOAD_FROM_PRETRAINED}_"
run_description += f"{datetime.datetime.now().strftime('%H_%M_%S')}"
print(f"========== {run_description} ===========")

CHECKPOINT_PATH = f"lightning_logs/{run_description}"
pretrain_checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_PATH,
    save_top_k=1,
    filename='pretrain-{epoch}',
    monitor='val_loss',
    mode='min'
)

checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_PATH,
    save_top_k=1,
    monitor='val_loss',
    mode='min'
)

# Save a copy of this file and configs file as a backup
save_copy_of_files(pretrain_checkpoint_callback)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if LOAD_FROM_PRETRAINED:
    best_model_path = pretrain_model()
else:
    best_model_path = ''

model, acc_results, f1_results = train_model(best_model_path)
print("ACC results", acc_results)
print("F1  results", f1_results)

# append result to a text file...
text_save_dir = "textFiles"
os.makedirs(text_save_dir, exist_ok=True)
f = open(f"{text_save_dir}/{MODEL_ID}.txt", 'a')
f.write(run_description + "  \n")
f.write(f"TSLANet_{os.path.basename(DATAPATH)}_l_{DEPTH}" + "  \n")
f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
f.write('\n')
f.write('\n')
f.close()