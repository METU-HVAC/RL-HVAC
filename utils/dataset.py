from datetime import datetime, timedelta
import random

def generate_chunks(start_date, days_per_chunk, total_days, step_size=10, seasons=['mixed']):
    """
    Generate date-based chunks for a given period, with seasons.

    Args:
        start_date (datetime): The starting date for generating chunks.
        days_per_chunk (int): Number of days in each chunk.
        total_days (int): Total number of days to divide into chunks.
        step_size (int): Step size for overlapping chunks.
        seasons (list of str): List of seasons to generate chunks for.

    Returns:
        list of tuple: A list of (chunk_start, chunk_end, season) tuples.
    """
    chunks = []

    for season in seasons:
        num_chunks = (total_days - days_per_chunk) // step_size + 1
        for i in range(num_chunks):
            chunk_start = start_date + timedelta(days=i * step_size)
            chunk_end = chunk_start + timedelta(days=days_per_chunk - 1)
            chunks.append((chunk_start, chunk_end, season))

    return chunks
# def split_chunks(chunks, train_ratio=0.6, val_ratio=0.3, seed=None):
#     """
#     Generate train chunks based on the train_ratio, and validation chunks as a subset of train chunks.

#     Args:
#         chunks (list of tuple): List of (chunk_start, chunk_end) tuples.
#         train_ratio (float): Proportion of chunks to be used for training.
#         val_ratio (float): Proportion of the training chunks to be used for validation.
#         seed (int, optional): Seed for random shuffling to ensure reproducibility.

#     Returns:
#         tuple: Train chunks, validation chunks (subset of train), and the remaining test chunks.
#     """
#     if seed is not None:
#         random.seed(seed)
#     random.shuffle(chunks)

#     train_size = int(train_ratio * len(chunks))
#     train_chunks = chunks[:train_size]

#     val_size = int(val_ratio * train_size)
#     val_chunks = train_chunks[:val_size]

#     test_chunks = chunks[train_size:]

#     return train_chunks, val_chunks, test_chunks
def split_chunks(chunks, train_ratio=0.7, val_ratio=0.1, seed=None):
    """
    Split chunks into train, validation, and test sets.

    Args:
        chunks (list of tuple): List of (chunk_start, chunk_end) tuples.
        train_ratio (float): Proportion of chunks for training.
        val_ratio (float): Proportion of chunks for validation.
        seed (int, optional): Seed for random shuffling to ensure reproducibility.

    Returns:
        tuple: Three lists containing train, validation, and test chunks.
    """
    if seed is not None:
        random.seed(seed)
    random.shuffle(chunks)

    train_size = int(train_ratio * len(chunks))
    val_size = int(val_ratio * len(chunks))
    test_size = len(chunks) - train_size - val_size

    train_chunks = chunks[:train_size]
    val_chunks = chunks[train_size:train_size + val_size]
    if test_size > 0:
        test_chunks = chunks[train_size + val_size:]
    else:
        test_chunks = []

    return train_chunks, val_chunks, test_chunks