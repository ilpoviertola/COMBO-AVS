from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

import torch
import os
import argparse
from pred import setup, Trainer, default_argument_parser


# dataset_dict["images"] = torch.randn(5, 3, 224, 224)
# dataset_dict["sem_segs"] = torch.randint(0, 2, (5, 1, 224, 224)).float()
# dataset_dict["audio_log_mel"] = torch.randn(5, 1, 96, 64)

def main():
    cfg = setup(args)
    if args.eval_only:
        best_ckpt_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
        print("Best checkpoint path: {}".format(best_ckpt_path))

    # Resolve a device from the accelerator (if available) or fall back to a safe torch device.
    accelerator = get_accelerator()
    device = None
    if accelerator is not None:
        # accelerator.device might be a callable or an attribute depending on implementation
        if callable(getattr(accelerator, "device", None)):
            try:
                device = accelerator.device(0)
            except TypeError:
                device = accelerator.device()
        elif hasattr(accelerator, "device"):
            device = accelerator.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Trainer.build_model(cfg).to(device)
    kwargs = {
        "batched_inputs":[{
            "images": torch.randn(5, 3, 224, 224).to(device),
            "sem_segs": torch.randint(0, 2, (5, 1, 224, 224)).float().to(device),
            "audio_log_mel": torch.randn(5, 1, 96, 64).to(device),
            "pre_masks": torch.randn(5, 1, 224, 224).to(device),
        }]
    }
    batch_size = 5
    flops, macs, params = get_model_profile(
        model=model,  # model
        input_shape=None,  # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args=[],  # list of positional arguments to the model.
        kwargs=kwargs,  # dictionary of keyword arguments to the model.
        print_profile=True,  # prints the model graph with the measured profile attached to each module
        detailed=True,  # print the detailed profile
        module_depth=-1,  # depth into the nested modules, with -1 being the inner most modules
        top_modules=1,  # the number of top modules to print aggregated profile
        warm_up=10,  # the number of warm-ups before measuring the time of each module
        as_string=True,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None,
    )  # the list of modules to ignore in the profiling
    print(
        f"Flops: {flops}, MACs: {macs}, Params: {params} for batch size {batch_size}"
    )


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main()
