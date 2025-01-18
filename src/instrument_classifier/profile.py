import torch
from torch.profiler import profile, ProfilerActivity, schedule

from instrument_classifier.train import main

def profile_training():
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.extend([ProfilerActivity.CUDA, ProfilerActivity.CUDA_MEMORY])

    def trace_handler(p):
        p.export_chrome_trace('trace.json')

    # A "step" typically corresponds to one batch of training data
    # The profiler schedule works as follows:
    # - wait=10: Skip profiling for the first 10 batches (letting the model stabilize)
    # - warmup=10: Profile the next 10 batches but discard the data (JIT warmup)
    # - active=5: Actually record profiling data for 5 batches
    # - repeat=6: Repeat this cycle 6 times to get multiple samples
    with profile(
        activities=activities,
        schedule=schedule(wait=10, warmup=10, active=5, repeat=6),
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_modules=True
    ) as prof:
        main()

if __name__ == "__main__":
    profile_training() 