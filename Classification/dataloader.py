import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import polars as pl


def normalize_time_series(data):
    mean = data.mean()
    std = data.std()
    normalized_data = (data - mean) / std
    return normalized_data


def zero_pad_sequence(input_tensor, pad_length):
    return torch.nn.functional.pad(input_tensor, (0, pad_length))


def calculate_padding(seq_len, patch_size):
    padding = patch_size - (seq_len % patch_size) if seq_len % patch_size != 0 else 0
    return padding


class Load_Dataset(torch.utils.data.Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_file):
        super(Load_Dataset, self).__init__()
        self.data_file = data_file

        # Load samples and labels
        x_data = data_file["samples"]  # dim: [#samples, #channels, Seq_len]

        # x_data = normalize_time_series(x_data)

        y_data = data_file.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data).squeeze()

        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data).squeeze()

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)

        self.x_data = x_data.to(torch.float32).squeeze()
        self.y_data = y_data.long().squeeze() if y_data is not None else None

        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index] if self.y_data is not None else None
        return x, y

    def __len__(self):
        return self.len


def get_datasets(DATASET_PATH, args):
    train_file = torch.load(os.path.join(DATASET_PATH, f"train.pt"))
    seq_len = train_file["samples"].shape[-1]
    required_padding = calculate_padding(seq_len, args.patch_size)

    val_file = torch.load(os.path.join(DATASET_PATH, f"val.pt"))
    test_file = torch.load(os.path.join(DATASET_PATH, f"test.pt"))

    train_dataset = Load_Dataset(train_file)
    val_dataset = Load_Dataset(val_file)
    test_dataset = Load_Dataset(test_file)

    if required_padding != 0:
        train_file["samples"] = zero_pad_sequence(train_file["samples"], required_padding)
        val_file["samples"] = zero_pad_sequence(val_file["samples"], required_padding)
        test_file["samples"] = zero_pad_sequence(test_file["samples"], required_padding)

    # in case the dataset is too small ...
    num_samples = train_dataset.x_data.shape[0]
    if num_samples < args.batch_size:
        batch_size = num_samples // 4
    else:
        batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader, test_loader



class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        features: list[str],
        targets: list[str],
        sequence_length: int = 5000,
        target_length: int = 1,
        step_size: int = 1,
        transform: callable = None,
        target_transform: callable = None
    ):
        """
        Time series dataset for PyTorch using Polars DataFrame.
        
        Args:
            df: Polars DataFrame containing the time series data
            features: List of feature column names
            targets: List of target column names
            sequence_length: Number of lagged samples (time steps) to include in each sequence
            target_length: Number of time steps to predict (default=1)
            step_size: Step size between sequences (default=1)
            transform: Optional transform to apply to features
            target_transform: Optional transform to apply to targets
        """
        self.df = df
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.target_length = target_length
        self.step_size = step_size
        self.transform = transform
        self.target_transform = target_transform
        
        # Convert relevant columns to numpy arrays
        self.feature_data = self.df.select(features).to_numpy()
        self.target_data = self.df.select(targets).to_numpy()
        
        # Calculate valid indices
        self.total_length = len(df)
        self.valid_indices = range(
            self.sequence_length - 1, 
            self.total_length - self.target_length, 
            self.step_size
        )
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get the index of the last element in the sequence
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.sequence_length + 1
        
        # Get the sequence of features
        sequence = self.feature_data[start_idx:end_idx + 1]
        
        # Get the target (can be multiple steps ahead)
        target_start = end_idx #+ 1
        target_end = target_start + self.target_length
        target = self.target_data[target_start:target_end]
                
        # Apply transforms if specified
        if self.transform:
            sequence = self.transform(sequence)
        if self.target_transform:
            target = self.target_transform(target)

        # Convert to torch tensors
        sequence = torch.from_numpy(sequence.copy()).to(torch.float32)
        target = torch.from_numpy(target.copy()).to(torch.int8)    

        sequence = sequence.permute(1, 0)  # [features, seq_len]

        return sequence, target


def create_timeseries_dataloader(
    df: pl.DataFrame,
    features: list[str],
    targets: list[str],
    sequence_length: int = 5000,
    target_length: int = 1,
    step_size: int = 1,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    transform: callable = None,
    target_transform: callable = None
) -> DataLoader:
    """
    Create a PyTorch DataLoader for time series data from a Polars DataFrame.
    
    Args:
        df: Polars DataFrame containing the time series data
        features: List of feature column names
        targets: List of target column names
        sequence_length: Number of lagged samples (time steps) to include in each sequence
        target_length: Number of time steps to predict (default=1)
        step_size: Step size between sequences (default=1)
        batch_size: Batch size for DataLoader (default=32)
        shuffle: Whether to shuffle the data (default=False)
        num_workers: Number of workers for DataLoader (default=0)
        transform: Optional transform to apply to features
        target_transform: Optional transform to apply to targets
        
    Returns:
        PyTorch DataLoader configured for time series data
    """
    dataset = TimeSeriesDataset(
        df=df,
        features=features,
        targets=targets,
        sequence_length=sequence_length,
        target_length=target_length,
        step_size=step_size,
        transform=transform,
        target_transform=target_transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


def get_file_paths(base_path, year, currency):
    """
    Retrieves all file paths matching a given pattern in a specified year directory.

    Parameters:
        base_path (str): Base directory path.
        year (str): Year folder name.
        pattern (str): File name pattern to match.

    Returns:
        list of str: List of file paths.
    """
    pattern=f"HISTDATA_COM_ASCII_{currency}_T*"
    search_path = os.path.join(base_path, currency, year, pattern)
    return glob.glob(search_path)



def get_csv_files_in_folder(folder_path):
    """
    Retrieves all CSV file paths in a specified folder.

    Parameters:
        folder_path (str): Path to the folder.

    Returns:
        list of str: List of file paths to CSV files.
    """
    # Use glob to match only `.csv` files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    return csv_files


def read_multiple_polar_csv(file_paths):
    """
    Reads multiple CSV files containing polar data and combines them into a single DataFrame.

    Parameters:
        file_paths (list of str): List of paths to the CSV files.

    Returns:
        pl.DataFrame: Combined DataFrame with parsed timestamps from all files.
    """
    data_frames = [read_polar_csv(file_path) for file_path in file_paths]
    return pl.concat(data_frames)


def read_polar_csv(file_path):
    """
    Reads a CSV file containing polar data (bid and ask) and processes it using the Polars library.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pl.DataFrame: Processed DataFrame with parsed timestamps.
    """
    # Define column names based on the structure of the file
    column_names = ["timestamp", "bid", "ask"]

    # Read the file into a Polars DataFrame
    df = pl.read_csv(file_path, has_header=False, new_columns=column_names, columns=[0,1,2])

    # Parse the timestamp into a datetime object
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y%m%d %H%M%S%f").alias("timestamp")
    )
    return df

def get_tick_datasets(base_path, year, currency, train_fraction=0.05, val_fraction=0.2, test_fraction=0.2, horizon=100):
    folder_paths = get_file_paths(base_path, str(year), currency)
    file_paths = [get_csv_files_in_folder(fol_path)[0] for fol_path in folder_paths]


    combined_df = read_multiple_polar_csv(file_paths)
    combined_df = combined_df.sort(by="timestamp")

    # Event description
    combined_df = combined_df.with_columns((combined_df["ask"] - combined_df["bid"]).alias("spread"))
    combined_df = combined_df.with_columns((combined_df["timestamp"] - combined_df["timestamp"].shift(1)).alias("delta"))
    combined_df = combined_df.with_columns(pl.arange(0, combined_df.height).alias("idx"))

    # # Event detection
    combined_df = combined_df.with_columns(((combined_df["bid"] + combined_df["ask"]) / 2).alias("mid"))

    combined_df = combined_df.with_columns(combined_df["mid"].rolling_mean(horizon).alias("m-"))
    combined_df = combined_df.with_columns(combined_df["mid"][::-1].rolling_mean(horizon)[::-1].alias("m+"))
    combined_df = combined_df.with_columns(((combined_df["m+"] - combined_df["m-"]) / combined_df["m-"]).alias("pc_change"))
    combined_df = combined_df.drop_nulls()

    combined_df = combined_df.with_columns(
            pl.when(pl.col("pc_change") <= -0.0002)
            .then(1)
            .when((pl.col("pc_change") > -0.0002) & (pl.col("pc_change") < 0.0002))
            .then(2)
            .otherwise(3)
            .alias("label"))

    combined_df = combined_df.with_columns(combined_df["label"].shift(-horizon).alias("shifted_label"))
    combined_df = combined_df.drop_nulls()
    combined_df = combined_df.sort(by="timestamp")

    features = ["bid", "delta"]
    target = ["shifted_label"]

    # reduce and sample dataframe
    combined_df = combined_df[:int(combined_df.shape[0] * train_fraction)]
    train_df, test_df = train_test_split(combined_df, train_size=1-test_fraction, shuffle=False)
    train_df, val_df = train_test_split(train_df, train_size=1-val_fraction, shuffle=False)

    # Create the dataloader
    train_dataloader = create_timeseries_dataloader(
        df=train_df,
        features=features,
        targets=target,
        sequence_length=200,  # 5000 lagged samples as requested
        target_length=1,      # Predict 1 step ahead
        step_size=1,          # Move window by 1 step each time
        batch_size=64,
        shuffle=True,
        transform=StandardScaler().fit_transform
    )

    val_dataloader = create_timeseries_dataloader(
        df=val_df,
        features=features,
        targets=target,
        sequence_length=200,  # 5000 lagged samples as requested
        target_length=1,      # Predict 1 step ahead
        step_size=1,          # Move window by 1 step each time
        batch_size=64,
        shuffle=False,
        transform=StandardScaler().fit_transform
    )

    test_dataloader = create_timeseries_dataloader(
        df=test_df,
        features=features,
        targets=target,
        sequence_length=500,  # 5000 lagged samples as requested
        target_length=1,      # Predict 1 step ahead
        step_size=1,          # Move window by 1 step each time
        batch_size=64,
        shuffle=False,
        transform=StandardScaler().fit_transform
    )

    return train_dataloader, val_dataloader, test_dataloader
