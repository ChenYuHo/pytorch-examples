import os
import torch
import torch.distributed as dist
from datetime import datetime
import tqdm
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
from time import perf_counter_ns
import torch.profiler
from torch.profiler import ExecutionTraceObserver

g_gigabyte = 1024**3

def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

def trace_handler(prof):
    kineto_file = f"worker_{dist.get_rank()}_step_{prof.step_num}"
    torch.profiler.tensorboard_trace_handler('./kineto', worker_name=kineto_file).__call__(prof)

def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
  
    if sampler:
        sampler.set_epoch(epoch)

    if args.profile_batches:
        et_file = f"t5_et_{dist.get_rank()}.json"
        et = ExecutionTraceObserver()
        et.register_callback(et_file)
        with torch.autograd.profiler.profile(
            use_kineto=True,
            record_shapes=True,
            with_flops=True,
            profile_memory=True,
            use_cuda=True
        ) as _:
            if rank == 0:
                print("Running dummy profiler warmup for CUPTI.")

        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=args.warmup_batches, active=args.profile_batches),
            record_shapes=True,
            with_flops=True,
            profile_memory=True,
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler
        ) as prof:
            num_batches = max(args.batches, args.warmup_batches + args.profile_batches)
            for iteration, batch in enumerate(train_loader):
                if iteration >= num_batches:
                    break
                tic = perf_counter_ns()
                if iteration == args.warmup_batches:
                    et.start()
                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)
                optimizer.zero_grad()
                output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
                loss = output["loss"]
                loss.backward()
                optimizer.step()
                fsdp_loss[0] += loss.item()
                fsdp_loss[1] += len(batch)
                if iteration == args.warmup_batches + args.profile_batches - 1:
                    et.stop()
                    et.unregister_callback()
                prof.step()
                if rank==0:
                    toc = perf_counter_ns()
                    print(f"iteration {iteration} elapsed {toc-tic} ns")

            dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
            train_accuracy = fsdp_loss[0] / fsdp_loss[1]
            return train_accuracy

    num_batches = args.batches + args.warmup_batches
    for iteration, batch in enumerate(train_loader):
        if num_batches and iteration >= num_batches:
            break
        tic = perf_counter_ns()
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank==0 and iteration >= args.warmup_batches:
            toc = perf_counter_ns()
            print(f"iteration {iteration} elapsed {toc-tic} ns")

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy


def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def setup_model(model_name):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer =  T5Tokenizer.from_pretrained(model_name)
        return model, tokenizer
