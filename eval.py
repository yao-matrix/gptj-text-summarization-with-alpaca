import os
import argparse
import numpy as np
import transformers

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset

import utils
import pickle

from accelerate import Accelerator

import evaluate
import nltk

from huggingface_hub import HfFolder

from tqdm import tqdm

nltk.download("punkt", quiet=False)

# Metric
metric = evaluate.load("rouge")
# evaluation generation args
gen_kwargs = {
    "early_stopping": True,
    "length_penalty": 2.0,
    "max_new_tokens": 128,
    "min_length": 30,
    "no_repeat_ngram_size": 3,
    "num_beams": 4,
}


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets

def get_memory_usage():
    memory_allocated = round(torch.cuda.memory_reserved()/1024**3, 3)

    print(f"GPU memory used total: {memory_allocated} GB")


def compute_metrics():

    torch.distributed.init_process_group(backend="nccl")

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()


    list_data_dict = utils.jload(data_args.data_path)

    logging.warning("Formatting inputs...")
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [prompt_input.format_map(example) for example in list_data_dict]
    targets = [f"{example['output']}" for example in list_data_dict]
    # print(f"sources: {sources}")
    print(f"source len: {len(sources)}")

    config = transformers.GPTJConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    rank = torch.distributed.get_rank()

    model_kwargs = { "low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16}
    model = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    device = torch.device(f"cuda:{rank}")
    model = model.eval().to(device)
    # get_memory_usage()
    model = model.to(memory_format=torch.channels_last)
    # get_memory_usage()

    preds = []
    with torch.inference_mode():
        src_len = len(sources)
        stride = src_len // torch.distributed.get_world_size()
        chunk_start = rank * stride
        step = 2
        chunk_end = min(src_len, (chunk_start + stride)) 
        for i in tqdm(range(chunk_start, chunk_end, step)):
            end = min(chunk_end, (i + step))
            input_batch = tokenizer.batch_encode_plus(sources[i:end], return_tensors="pt", 
                                                      padding=True, truncation=True, 
                                                      max_length=1919)

            # print(f"input_tokens: {input_tokens}")
            for t in input_batch:
                if torch.is_tensor(input_batch[t]):
                    input_batch[t] = input_batch[t].to(device)

            output_batch = model.generate(**input_batch, **gen_kwargs)
            # print(f"outputs: {outputs[0]}")

            input_batch_lengths = [x.shape[0] for x in input_batch.input_ids]
            # print(outputs)
            output_batch_lengths = [x.shape[0] for x in output_batch]

            output_batch_truncated=[]
            for data, source_len in zip(output_batch, input_batch_lengths):
                output_batch_truncated.append(data[source_len:])
            # print(f"outputs_trincated: {outputs_truncated[0]}")

            pred_batch = tokenizer.batch_decode(output_batch_truncated, skip_special_tokens=True)

            preds.extend(pred_batch)

        # gather preds to rank 0
        # prediction_list = [None for _ in range(torch.distributed.get_world_size())]
        # preds = torch.tensor(preds)
        # if rank == 0:
        #     torch.distributed.gather(preds, prediction_list)
        # else:
        #     torch.distributed.gather(preds)

        # if rank == 0:
        # preds = [j for sub in prediction_list for j in sub]

        # Some simple post-processing
        preds, targets = postprocess_text(preds, targets[chunk_start:chunk_end])

        # print(f"preds: {preds[0]}")
        # print(f"targets: {targets[0]}")

        # accelerator = Accelerator()

        result = metric.compute(predictions=preds, references=targets, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [len(pred) for pred in preds]
        result["gen_len"] = np.sum(prediction_lens)
        result["gen_num"] = len(preds)

        # gathered_results = accelerator.gather(result)

        print(result)
        # print(gathered_results)

if __name__ == "__main__":
    compute_metrics()
