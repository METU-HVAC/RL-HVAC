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


from collections import defaultdict

def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "fall"

def stratified_train_val_split(chunks, train_ratio=0.8, val_ratio=0.2, seed=None):
    """
    Stratified split of chunks into train and validation sets by season.

    Args:
        chunks (list of tuple): List of (chunk_start, chunk_end) tuples.
                                chunk_start must have .month or be a (month, day) tuple.
        train_ratio (float): Proportion of chunks for training.
        val_ratio (float): Proportion of chunks for validation.
        seed (int, optional): Seed for shuffling.

    Returns:
        tuple: (train_chunks, val_chunks)
    """
    assert 0 < train_ratio + val_ratio <= 1.0, "Train + Val ratio must be between 0 and 1"

    if seed is not None:
        random.seed(seed)

    season_buckets = defaultdict(list)
    for chunk in chunks:
        chunk_start = chunk[0]
        month = chunk_start.month if hasattr(chunk_start, 'month') else chunk_start[0]
        season = get_season(month)
        season_buckets[season].append(chunk)

    train_chunks, val_chunks,test_chunks = [], [],[]

    for season, season_chunks in season_buckets.items():
        random.shuffle(season_chunks)
        total = len(season_chunks)
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)

        train_chunks += season_chunks[:train_end]
        val_chunks += season_chunks[train_end:val_end]

    return train_chunks, val_chunks, test_chunks

def season_balanced_split_fixed_val_count(chunks, val_chunk_count=8, train_ratio=0.8, seed=None):
    """
    Split into train and validation sets ensuring a fixed number of validation chunks,
    distributed across seasons.

    Args:
        chunks (list): List of (chunk_start, chunk_end) tuples.
        val_chunk_count (int): Total number of validation chunks desired.
        train_ratio (float): Ratio of training from the remaining pool.
        seed (int): Seed for reproducibility.

    Returns:
        train_chunks, val_chunks
    """
    if seed is not None:
        random.seed(seed)

    from collections import defaultdict
    import math

    def get_season(month):
        if month in [12, 1, 2]: return "winter"
        elif month in [3, 4, 5]: return "spring"
        elif month in [6, 7, 8]: return "summer"
        elif month in [9, 10, 11]: return "fall"

    # Group chunks by season
    season_buckets = defaultdict(list)
    for chunk in chunks:
        month = chunk[0].month if hasattr(chunk[0], 'month') else chunk[0][0]
        season_buckets[get_season(month)].append(chunk)

    # Determine per-season val chunk allocation (rounding to nearest)
    season_keys = list(season_buckets.keys())
    val_chunks = []
    val_alloc = {season: 0 for season in season_keys}

    # Allocate validation chunks proportionally to available chunks per season
    total_chunks = sum(len(v) for v in season_buckets.values())
    for season in season_keys:
        frac = len(season_buckets[season]) / total_chunks
        val_alloc[season] = round(frac * val_chunk_count)

    # Adjust to hit exact val_chunk_count (in case of rounding mismatch)
    total_alloc = sum(val_alloc.values())
    while total_alloc != val_chunk_count:
        # Add or remove from largest bucket
        key = max(val_alloc, key=lambda k: val_alloc[k])
        if total_alloc > val_chunk_count and val_alloc[key] > 0:
            val_alloc[key] -= 1
        elif total_alloc < val_chunk_count:
            val_alloc[key] += 1
        total_alloc = sum(val_alloc.values())

    train_chunks = []

    for season in season_keys:
        chunks_in_season = season_buckets[season]
        random.shuffle(chunks_in_season)
        val_count = val_alloc[season]
        val_chunks.extend(chunks_in_season[:val_count])
        remaining = chunks_in_season[val_count:]
        train_cutoff = int(train_ratio * len(remaining))
        train_chunks.extend(remaining[:train_cutoff])

    return train_chunks, val_chunks

from collections import defaultdict
import random

def balanced_month_sample(chunks, val_chunks_per_month=2, seed=None):
    month_dict = defaultdict(list)
    for start, end, season in chunks:
        month = start.month
        month_dict[month].append((start, end, season))

    val_chunks = []
    remaining_chunks = []
    for month, items in month_dict.items():
        if seed is not None:
            random.seed(seed + month)
        random.shuffle(items)
        val_chunks.extend(items[:val_chunks_per_month])
        remaining_chunks.extend(items[val_chunks_per_month:])

    random.shuffle(remaining_chunks)
    train_chunks = remaining_chunks

    return train_chunks, val_chunks, []