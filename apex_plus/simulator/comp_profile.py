from functools import lru_cache
from typing import List

import pandas as pd

from apex_plus.ir.tasks.attention import MHAHead
from apex_plus.ir.tasks.ffn import MLPFilter, GLUFilter, SwiGLUFilter
from apex_plus.utils.dtype import DTYPE, dtype_to_str
from apex_plus.utils.logger import debug_log, warning_log

GEMM_TMPL = "profile/comp/{gpu}/gemm_{freq}.csv"  # e.g. gemm_1980.csv
MHA_TMPL = "profile/comp/{gpu}/mha_{freq}.csv"
BI_MHA_TMPL = "profile/comp/{gpu}/bimha_{freq}.csv"


@lru_cache(maxsize=512)
def _load_table(op_kind: str, gpu: str, freq: int) -> pd.DataFrame:
    """
    op_kind ∈ {"gemm", "mha", "bimha"}
    """
    name = {"gemm": GEMM_TMPL, "mha": MHA_TMPL, "bimha": BI_MHA_TMPL}[op_kind].format(
        gpu = gpu,
        freq=freq
    )
    df = pd.read_csv(name)

    return df


def _gemm_df(gpu: str, freq: int) -> pd.DataFrame:
    return _load_table("gemm", gpu, freq)


def _mha_df(gpu: str, freq: int) -> pd.DataFrame:
    return _load_table("mha", gpu, freq)


def _bimha_df(gpu: str, freq: int) -> pd.DataFrame:
    return _load_table("bimha", gpu, freq)


def _interpolate(
    df: pd.DataFrame,
    col1: str,
    col1_val: int,
    col2: str,
    col2_val: int,
    target_col: str,
) -> float:
    small = df[(df[col1] <= col1_val) & (df[col2] <= col2_val)]
    large = df[(df[col1] >= col1_val) & (df[col2] >= col2_val)]
    if len(small) == 0 or len(large) == 0:
        if len(small) == 0 and len(large) != 0:
            return large.iloc[0][target_col]
        if len(large) == 0 and len(small) != 0:
            return small.iloc[-1][target_col]
        else:
            # Both empty - try to find closest points for interpolation
            # First, try to find points with same col1, interpolate on col2
            df_col1 = df[df[col1] == col1_val]
            if len(df_col1) > 0:
                # Interpolate on col2 only
                small_col2 = df_col1[df_col1[col2] <= col2_val]
                large_col2 = df_col1[df_col1[col2] >= col2_val]
                if len(small_col2) > 0 and len(large_col2) > 0:
                    small_val = small_col2.iloc[-1][target_col]
                    large_val = large_col2.iloc[0][target_col]
                    small_col2_val = small_col2.iloc[-1][col2]
                    large_col2_val = large_col2.iloc[0][col2]
                    if small_col2_val == large_col2_val:
                        return small_val
                    r = (col2_val - small_col2_val) / (large_col2_val - small_col2_val)
                    return small_val * (1 - r) + large_val * r
                elif len(small_col2) > 0:
                    return small_col2.iloc[-1][target_col]
                elif len(large_col2) > 0:
                    return large_col2.iloc[0][target_col]
            
            # Second, try to find points with same col2, interpolate on col1
            df_col2 = df[df[col2] == col2_val]
            if len(df_col2) > 0:
                # Interpolate on col1 only
                small_col1 = df_col2[df_col2[col1] <= col1_val]
                large_col1 = df_col2[df_col2[col1] >= col1_val]
                if len(small_col1) > 0 and len(large_col1) > 0:
                    small_val = small_col1.iloc[-1][target_col]
                    large_val = large_col1.iloc[0][target_col]
                    small_col1_val = small_col1.iloc[-1][col1]
                    large_col1_val = large_col1.iloc[0][col1]
                    if small_col1_val == large_col1_val:
                        return small_val
                    r = (col1_val - small_col1_val) / (large_col1_val - small_col1_val)
                    return small_val * (1 - r) + large_val * r
                elif len(small_col1) > 0:
                    return small_col1.iloc[-1][target_col]
                elif len(large_col1) > 0:
                    return large_col1.iloc[0][target_col]
            
            # Third, try to interpolate on col2 with closest col1 values
            # Find closest col1 values
            unique_col1 = sorted(df[col1].unique())
            if len(unique_col1) >= 2:
                # Find col1 values that bracket col1_val
                small_col1_vals = [v for v in unique_col1 if v <= col1_val]
                large_col1_vals = [v for v in unique_col1 if v >= col1_val]
                if len(small_col1_vals) > 0 and len(large_col1_vals) > 0:
                    small_col1_val = max(small_col1_vals)
                    large_col1_val = min(large_col1_vals)
                    # Get data for these col1 values and interpolate on col2
                    df_small_col1 = df[df[col1] == small_col1_val]
                    df_large_col1 = df[df[col1] == large_col1_val]
                    small_col2 = df_small_col1[df_small_col1[col2] <= col2_val]
                    large_col2 = df_small_col1[df_small_col1[col2] >= col2_val]
                    if len(small_col2) > 0 and len(large_col2) > 0:
                        val1 = small_col2.iloc[-1][target_col] * (1 - (col1_val - small_col1_val) / (large_col1_val - small_col1_val)) if small_col1_val != large_col1_val else small_col2.iloc[-1][target_col]
                        val2 = large_col2.iloc[0][target_col] * ((col1_val - small_col1_val) / (large_col1_val - small_col1_val)) if small_col1_val != large_col1_val else large_col2.iloc[0][target_col]
                        # Interpolate on col2
                        r2 = (col2_val - small_col2.iloc[-1][col2]) / (large_col2.iloc[0][col2] - small_col2.iloc[-1][col2])
                        return val1 * (1 - r2) + val2 * r2
            
            # Last resort: find closest point by Euclidean distance
            if len(df) > 0:
                df['distance'] = ((df[col1] - col1_val) ** 2 + (df[col2] - col2_val) ** 2) ** 0.5
                closest = df.nsmallest(1, 'distance').iloc[0]
                return closest[target_col]
            
            raise ValueError(
                "Cannot interpolate. "
                f"col1: {col1}, col1_val: {col1_val}, "
                f"col2: {col2}, col2_val: {col2_val}."
            )

    small = small.iloc[-1]
    large = large.iloc[0]
    if small[col1] == large[col1] and small[col2] == large[col2]:
        return small[target_col]
    elif small[col1] == large[col1]:
        r2 = (col2_val - small[col2]) / (large[col2] - small[col2])
        return small[target_col] * (1 - r2) + large[target_col] * r2
    elif small[col2] == large[col2]:
        r1 = (col1_val - small[col1]) / (large[col1] - small[col1])
        return small[target_col] * (1 - r1) + large[target_col] * r1
    else:
        r1 = (col1_val - small[col1]) / (large[col1] - small[col1])
        r2 = (col2_val - small[col2]) / (large[col2] - small[col2])
        r = (r1 * r2) ** 0.5
        return small[target_col] * (1 - r) + large[target_col] * r


@lru_cache(maxsize=512)
def _gemm_time(
    gpu: str,
    frequency: int,
    m: int,
    k: int,
    n: int,
    dtype: str,
) -> float:
    df = _gemm_df(gpu,frequency)
    df = df[df["dtype"] == dtype]
    
    # Check if n exceeds the profiled range
    available_n_values = sorted(df["n"].unique())
    max_profiled_n = max(available_n_values) if available_n_values else n
    
    if n > max_profiled_n:
        # Use linear extrapolation: find data at max_profiled_n and scale by n ratio
        df_max_n = df[df["n"] == max_profiled_n]
        assert (
            not df_max_n.empty
        ), f"Cannot find gemm time for {gpu}, freq={frequency}, {dtype}, n={max_profiled_n}"
        
        # Get time at max_profiled_n and scale linearly by n ratio
        time_at_max = _interpolate(df_max_n, "m", m, "k", k, "time(us)")
        exe_time = time_at_max * (n / max_profiled_n)
        debug_log(f"[TIME] GEMM extrapolation: n={n} > max_profiled_n={max_profiled_n}, "
                  f"time_at_max={time_at_max:.2f}us, exe_time={exe_time:.2f}us")
        if exe_time < 0:
            warning_log(f"[⚠️ALERT] NEGATIVE TIME detected in GEMM extrapolation! exe_time={exe_time:.2f}us")
        
        # Energy extrapolation (same approach)
        if frequency != 0:
            energy_at_max = _interpolate(df_max_n, "m", m, "k", k, "avg_energy(uJ)")
            exe_energy = energy_at_max * (n / max_profiled_n)
            debug_log(f"[ENERGY] GEMM extrapolation: n={n} > max_profiled_n={max_profiled_n}, "
                      f"energy_at_max={energy_at_max:.2f}uJ, exe_energy={exe_energy:.2f}uJ")
            if exe_energy < 0:
                warning_log(f"[⚠️ALERT] NEGATIVE ENERGY detected in GEMM extrapolation! exe_energy={exe_energy:.2f}uJ")
        else:
            exe_energy = 0
    else:
        # Normal case: n is within profiled range
        df = df[df["n"] == n]
        assert (
            not df.empty
        ), f"Cannot find gemm time for {gpu}, freq={frequency}, {dtype},{m},{k},{n}"
        exe_time = _interpolate(df, "m", m, "k", k, "time(us)")
        debug_log(f"[TIME] GEMM interpolation: m={m}, k={k}, n={n}, frequency={frequency}, "
                  f"exe_time={exe_time:.2f}us")
        if exe_time < 0:
            warning_log(f"[⚠️ALERT] NEGATIVE TIME detected in GEMM interpolation! exe_time={exe_time:.2f}us")
        # Get energy consumption if profiling exist; otherwise return energy = 0
        if frequency != 0:
            exe_energy = _interpolate(df, "m", m, "k", k, "avg_energy(uJ)")
            debug_log(f"[ENERGY] GEMM interpolation: m={m}, k={k}, n={n}, frequency={frequency}, "
                      f"exe_energy={exe_energy:.2f}uJ")
            if exe_energy < 0:
                warning_log(f"[⚠️ALERT] NEGATIVE ENERGY detected in GEMM interpolation! exe_energy={exe_energy:.2f}uJ")
        else:
            exe_energy = 0
    
    return exe_time, exe_energy


def round_to_power_of_2(n):
    # If n is already a power of 2, return n
    if (n & (n - 1)) == 0:
        return n
    # Find the closest power of 2 greater than or equal to n
    power_of_2_greater = 1
    while power_of_2_greater < n:
        power_of_2_greater <<= 1
    # Find the closest power of 2 less than n
    power_of_2_less = power_of_2_greater >> 1
    # Return the closest power of 2
    if (n - power_of_2_less) < (power_of_2_greater - n):
        return power_of_2_less
    else:
        return power_of_2_greater


def gemm_time(
    gpu: str,
    frequency: int,
    m: int,
    k: int,
    n: int,
    dtype: str,
) -> float:
    # Round up to the nearest multiple of 128.

    n = (n + 127) // 128 * 128 if n > 64 else 64

    if dtype == "float8":
        m = 16 if m < 16 else m
        k = 16 if k < 16 else k
        m = round_to_power_of_2(m)
        k = round_to_power_of_2(k)
        n = round_to_power_of_2(n)
    return _gemm_time(gpu, frequency, m, k, n, dtype)


@lru_cache(maxsize=512)
def attn_time(
    gpu: str,
    frequency: int,
    head_size: int,
    num_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: str,
) -> float:
    df = _mha_df(gpu,frequency)
    df = df[df["dtype"] == dtype]
    df = df[df["head_size"] == head_size]
    assert (
        not df.empty
    ), f"Cannot find attn time for {gpu}, {dtype}, {head_size}, {frequency}"
    # Round up to the nearest multiple of 16.
    seq_len = (seq_len + 15) // 16 * 16
    # If seq_len exceeds max in profiling data, use the maximum available
    max_seq_len = df["seq_len"].max()
    if seq_len > max_seq_len:
        seq_len = max_seq_len
    df = df[df["seq_len"] == seq_len]

    exe_time = _interpolate(
        df, "batch_size", batch_size, "num_heads", num_heads, "time(us)"
    )
    exe_energy = (
        _interpolate(
            df, "batch_size", batch_size, "num_heads", num_heads, "avg_energy(uJ)"
        )
        if frequency != 0
        else 0
    )
    return exe_time, exe_energy


@lru_cache(maxsize=512)
def bi_attn_time(
    gpu: str,
    frequency: int,
    head_size: int,
    num_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: str,
) -> float:
    df = _bimha_df(gpu,frequency)
    df = df[df["dtype"] == dtype]
    df = df[df["head_size"] == head_size]
    assert (
        not df.empty
    ), f"Cannot find attn time for {gpu}, {dtype}, {head_size}, {frequency}"
    # Round up to the nearest multiple of 16.
    seq_len = (seq_len + 15) // 16 * 16
    df = df[df["seq_len"] == seq_len]
    exe_time = _interpolate(
        df, "batch_size", batch_size, "num_heads", num_heads, "time(us)"
    )
    exe_energy = (
        _interpolate(
            df, "batch_size", batch_size, "num_heads", num_heads, "avg_energy(uJ)"
        )
        if frequency != 0
        else 0
    )
    return exe_time, exe_energy


def mha_time(
    gpu: str,
    frequency: int,
    heads,  # can be List[MHAHead] or List[BiMHAHead]
    dtype: DTYPE,
    input_lens: List[int],
    cached_lens: List[int],
    prefix_cached_tokens: List[int],
    masked: bool,  # True for MHA and False for BiMHA
) -> float:
    if not heads:
        return 0.0
    if not input_lens:
        return 0.0

    # Ensure prefix_cached_tokens is aligned with input_lens
    if len(prefix_cached_tokens) != len(input_lens):
        # Pad with zeros if shorter, truncate if longer
        if len(prefix_cached_tokens) < len(input_lens):
            prefix_cached_tokens = prefix_cached_tokens + [0] * (len(input_lens) - len(prefix_cached_tokens))
        else:
            prefix_cached_tokens = prefix_cached_tokens[:len(input_lens)]

    dtype_str = dtype_to_str(dtype)
    # Synthetic results for attention for MHA in FP8 (temporary)
    # Synthetic FP8 = Result of FP16 divided by 1.5
    # Future plan: add real profiling for MHA in FP8
    atten_dtype = "half" if dtype == DTYPE.FLOAT8 else dtype_to_str(dtype)
    num_heads = len(heads)
    head_size = heads[0].head_size
    hidden_size = heads[0].hidden_size

    num_total_input_tokens = sum(input_lens)
    total_time = 0.0
    total_energy = 0.0
    
    # 1. QKV Linear - split computation for cached vs non-cached tokens
    # For cached tokens: only compute Q (1/3 of QKV computation)
    # For non-cached tokens: compute full QKV
    total_cached_tokens = sum(prefix_cached_tokens)
    total_non_cached_tokens = num_total_input_tokens - total_cached_tokens
    
    # Compute QKV for non-cached tokens (full QKV)
    if total_non_cached_tokens > 0:
        exe_time, exe_energy = gemm_time(
            gpu=gpu,
            frequency=frequency,
            m=3 * num_heads * head_size,
            k=hidden_size,
            n=total_non_cached_tokens,
            dtype=dtype_str,
        )
        total_time += exe_time
        total_energy += exe_energy
    
    # Compute only Q for cached tokens (1/3 of QKV computation)
    if total_cached_tokens > 0:
        exe_time, exe_energy = gemm_time(
            gpu=gpu,
            frequency=frequency,
            m=num_heads * head_size,  # Only Q, not QKV
            k=hidden_size,
            n=total_cached_tokens,
            dtype=dtype_str,
        )
        total_time += exe_time
        total_energy += exe_energy

    # 2. Attention
    prompt_indices = [i for i in range(len(input_lens)) if cached_lens[i] == 0]
    if prompt_indices:
        prompt_batch_size = len(prompt_indices)
        # Attention uses FULL sequence length (cached + non-cached) because
        # Q needs to attend to the entire KV cache (cached KV + new KV)
        total_prompt_tokens = sum(input_lens[i] for i in prompt_indices)
        prompt_seq_len = total_prompt_tokens / prompt_batch_size
        
        # Only compute attention if there are tokens to process
        if prompt_seq_len > 0:
            if masked:
                attention_time, attention_energy = attn_time(
                    gpu=gpu,
                    frequency=frequency,
                    head_size=head_size,
                    num_heads=num_heads,
                    batch_size=prompt_batch_size,
                    seq_len=prompt_seq_len,
                    dtype=atten_dtype,
                )
            else:
                attention_time, attention_energy = bi_attn_time(
                    gpu=gpu,
                    frequency=frequency,
                    head_size=head_size,
                    num_heads=num_heads,
                    batch_size=prompt_batch_size,
                    seq_len=prompt_seq_len,
                    dtype=atten_dtype,
                )
            # Synthetic results for attention for MHA in FP8 (temporary)
            # Synethetic FP8 = Result of FP16/1.5
            attention_time = (
                (attention_time / 1.5) if dtype == DTYPE.FLOAT8 else attention_time
            )
            total_time += attention_time
            total_energy += attention_energy

    # 3. Cached Attention
    decoding_indices = [i for i in range(len(input_lens)) if cached_lens[i] > 0]
    if decoding_indices:
        decoding_batch_size = len(decoding_indices)
        decoding_seq_len = 1
        exe_time, exe_energy = attn_time(
            gpu=gpu,
            frequency=frequency,
            head_size=head_size,
            num_heads=num_heads,
            batch_size=decoding_batch_size,
            seq_len=decoding_seq_len,
            dtype=atten_dtype,
        )
        total_time += exe_time
        total_energy += exe_energy

    # 4. Output Linear
    exe_time, exe_energy = gemm_time(
        gpu=gpu,
        frequency=frequency,
        m=hidden_size,
        k=num_heads * head_size,
        n=num_total_input_tokens,
        dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy
    return total_time, total_energy


def mlp_time(
    gpu: str,
    frequency: int,
    filters: List[MLPFilter],
    dtype: DTYPE,
    num_tokens: int,
) -> float:
    if not filters:
        return 0.0

    num_filters = len(filters)
    hidden_size = filters[0].hidden_size
    dtype_str = dtype_to_str(dtype)
    total_time = 0.0
    total_energy = 0.0

    # 1. MLP 0
    total_time, total_energy = gemm_time(
        gpu=gpu,
        frequency=frequency,
        m=num_filters,
        k=hidden_size,
        n=num_tokens,
        dtype=dtype_str,
    )

    # 2. MLP 1
    exe_time, exe_energy = gemm_time(
        gpu=gpu,
        frequency=frequency,
        m=hidden_size,
        k=num_filters,
        n=num_tokens,
        dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy
    return total_time, total_energy


def glu_time(
    gpu: str,
    frequency: int,
    filters: List[GLUFilter],
    dtype: DTYPE,
    num_tokens: int,
) -> float:
    if not filters:
        return 0.0

    num_filters = len(filters)
    hidden_size = filters[0].hidden_size
    dtype_str = dtype_to_str(dtype)
    total_time = 0.0
    total_energy = 0.0

    # 1. MLP 0
    total_time, total_energy = gemm_time(
        gpu=gpu,
        frequency=frequency,
        m=num_filters * 2,
        k=hidden_size,
        n=num_tokens,
        dtype=dtype_str,
    )

    # 2. MLP 1
    exe_time, exe_energy = gemm_time(
        gpu=gpu,
        frequency=frequency,
        m=hidden_size,
        k=num_filters,
        n=num_tokens,
        dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy
    return total_time, total_energy


def swiglu_time(
    gpu: str,
    frequency: int,
    filters,
    dtype: DTYPE,
    num_tokens: int,
) -> float:
    if not filters:
        return 0.0

    num_filters = len(filters)
    hidden_size = filters[0].hidden_size
    dtype_str = dtype_to_str(dtype)
    total_time = 0.0
    total_energy = 0.0

    # 1. MLP 0
    total_time, total_energy = gemm_time(
        gpu=gpu,
        frequency=frequency,
        m=num_filters,
        k=hidden_size,
        n=num_tokens,
        dtype=dtype_str,
    )

    # 2. MLP 1
    exe_time, exe_energy = gemm_time(
        gpu=gpu,
        frequency=frequency,
        m=hidden_size,
        k=num_filters,
        n=num_tokens,
        dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy

    # 3. MLP 2
    exe_time, exe_energy = gemm_time(
        gpu=gpu,
        frequency=frequency,
        m=num_filters,
        k=hidden_size,
        n=num_tokens,
        dtype=dtype_str,
    )
    total_time += exe_time
    total_energy += exe_energy
    return total_time, total_energy
