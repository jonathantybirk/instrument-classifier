"""Module for profiling the training process of the instrument classifier.

This module provides functionality to profile the model training process,
analyzing CPU/GPU usage, memory consumption, and operation performance.
"""

import torch
from torch.profiler import profile, ProfilerActivity, schedule

from train import train_model


def print_profiler_stats(prof: profile) -> None:
    """Print key statistics from the profiling results.

    Args:
        prof: Profiler object containing the profiling data
    """
    print("\n=== Profiling Statistics ===")
    
    # Print overall stats
    print("\nTop 10 time-consuming operations:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    # Memory stats if available
    if torch.cuda.is_available():
        print("\nGPU Memory Stats:")
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    
    print("\nOperation Statistics by Input Shape:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def profile_training() -> None:
    """Profile the model training process.
    
    This function sets up the PyTorch profiler to analyze the training process.
    It captures:
    - CPU/GPU activity
    - Memory usage
    - Operation shapes
    - Stack traces
    
    The profiling results are:
    1. Exported as a Chrome trace file ('trace.json')
    2. Printed as summary statistics to the console
    """
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.extend([ProfilerActivity.CUDA, ProfilerActivity.CUDA_MEMORY])

    def trace_handler(p: profile) -> None:
        """Stop training after profiling is complete, export trace and print stats."""
        p.export_chrome_trace('trace.json')
        # Calculate total steps based on schedule parameters
        total_steps = (10 + 10 + 5) * 10  # (wait + warmup + active) * repeat
        if p.step_num >= total_steps - 1:
            print("\nProfiling completed!")
            print_profiler_stats(p)
            # Signal to stop training
            raise StopIteration("Profiling completed")

    # A "step" corresponds to one batch of training data
    # The profiler schedule works as follows:
    # - wait=10: Skip profiling for the first 10 batches (letting the model stabilize)
    # - warmup=10: Profile the next 10 batches but discard the data (JIT warmup)
    # - active=5: Actually record profiling data for 5 batches
    # - repeat=10: Repeat this cycle 10 times to get multiple samples
    try:
        with profile(
            activities=activities,
            schedule=schedule(wait=10, warmup=10, active=5, repeat=10),
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_modules=True
        ) as prof:
            try:
                train_model(profiler=prof)
            except StopIteration as e:
                if str(e) != "Profiling completed":
                    raise
                print("Training stopped after profiling completion")
            
        # Print statistics after training
        print_profiler_stats(prof)
    except Exception as e:
        if not isinstance(e, StopIteration):
            print(f"Error during profiling: {str(e)}")
            raise


if __name__ == "__main__":
    profile_training() 