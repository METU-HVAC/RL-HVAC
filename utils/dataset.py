from datetime import datetime, timedelta
import random

def generate_chunks(start_date, days_per_chunk, total_days):
    """
    Generate date-based chunks for a given period.

    Args:
        start_date (datetime): The starting date for generating chunks.
        days_per_chunk (int): Number of days in each chunk.
        total_days (int): Total number of days to divide into chunks.

    Returns:
        list of tuple: A list of (chunk_start, chunk_end) tuples.
    """
    chunks = []
    for i in range(total_days // days_per_chunk):
        chunk_start = start_date + timedelta(days=i * days_per_chunk)
        chunk_end = chunk_start + timedelta(days=days_per_chunk - 1)
        chunks.append((chunk_start, chunk_end))
    return chunks

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

    train_chunks = chunks[:train_size]
    val_chunks = chunks[train_size:train_size + val_size]
    test_chunks = chunks[train_size + val_size:]

    return train_chunks, val_chunks, test_chunks