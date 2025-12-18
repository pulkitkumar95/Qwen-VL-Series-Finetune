import os
import torch
from peft import PeftModel
import json
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import (
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig, 
    Qwen2VLForConditionalGeneration, 
    HfArgumentParser, 
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from src.dataset import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from monkey_patch_forward import (
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward, 
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward
)
from monkey_patch_vision import replace_qwen2_5_vision
from transformers import Trainer
import numpy as np
from typing import Dict, List, Optional

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

@dataclass
class EvalArguments(TrainingArguments):
    """Arguments for evaluation."""
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save evaluation results (JSON format)."}
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate."}
    )
    do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use sampling for generation."}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for generation."}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p for generation."}
    )
    top_k: int = field(
        default=50,
        metadata={"help": "Top-k for generation."}
    )
    compute_perplexity: bool = field(
        default=True,
        metadata={"help": "Whether to compute perplexity."}
    )
    save_predictions: bool = field(
        default=True,
        metadata={"help": "Whether to save model predictions."}
    )

def evaluate():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, EvalArguments))
    
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    
    if data_args.eval_path is None:
        raise ValueError("`eval_path` must be provided for evaluation.")
    
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    local_rank = eval_args.local_rank
    compute_dtype = (torch.float16 if eval_args.fp16 else (torch.bfloat16 if eval_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if eval_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":eval_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=eval_args.bits==4,
                load_in_8bit=eval_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=eval_args.double_quant,
                bnb_4bit_quant_type=eval_args.quant_type,
            )
        ))

    config = AutoConfig.from_pretrained(model_args.model_id)

    # Load model with appropriate monkey patches
    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not eval_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )
    elif config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not eval_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )
    elif config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not eval_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )
    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not eval_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )

    # Load LoRA weights if specified
    if eval_args.lora_enable and eval_args.lora_weight_path:
        rank0_print(f"Loading LoRA weights from {eval_args.lora_weight_path}")
        model = PeftModel.from_pretrained(model, eval_args.lora_weight_path)
        if eval_args.bits == 16:
            if eval_args.bf16:
                model = model.to(torch.bfloat16)
            if eval_args.fp16:
                model = model.to(torch.float16)

    model.config.use_cache = True
    model.eval()

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # Prepare evaluation dataset
    eval_dataset = make_supervised_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args
    )["eval_dataset"]

    if eval_dataset is None:
        raise ValueError("Evaluation dataset is None. Please check `eval_path`.")

    rank0_print(f"Evaluating on {len(eval_dataset)} examples")

    # Create a simple trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
    )

    # Compute perplexity if requested
    metrics = {}
    if eval_args.compute_perplexity:
        rank0_print("Computing perplexity...")
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        metrics.update(eval_results)
        eval_loss = eval_results.get('eval_loss', None)
        if eval_loss is not None:
            perplexity = torch.exp(torch.tensor(eval_loss)).item()
            metrics['perplexity'] = perplexity
            rank0_print(f"Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")
        else:
            rank0_print("Could not compute perplexity (loss not available)")

    # Generate predictions
    predictions = []
    
    if eval_args.save_predictions:
        rank0_print("Generating predictions...")
        device = next(model.parameters()).device
        
        for i in tqdm(range(len(eval_dataset)), desc="Generating"):
            example = eval_dataset[i]
            
            # Prepare inputs
            inputs = {}
            for key, value in example.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.unsqueeze(0).to(device)
                else:
                    inputs[key] = value
            
            # Extract prompt (everything before the response)
            input_ids = inputs["input_ids"]
            labels = inputs.get("labels", None)
            
            # Find where the response starts (first non-IGNORE_INDEX label)
            if labels is not None:
                response_start_idx = (labels != -100).nonzero(as_tuple=True)[0]
                if len(response_start_idx) > 0:
                    prompt_length = response_start_idx[0].item()
                else:
                    prompt_length = input_ids.shape[-1]
            else:
                prompt_length = input_ids.shape[-1]
            
            prompt_ids = input_ids[:, :prompt_length]
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    prompt_ids,
                    attention_mask=inputs.get("attention_mask", None)[:, :prompt_length] if "attention_mask" in inputs else None,
                    pixel_values=inputs.get("pixel_values", None),
                    image_grid_thw=inputs.get("image_grid_thw", None),
                    pixel_values_videos=inputs.get("pixel_values_videos", None),
                    video_grid_thw=inputs.get("video_grid_thw", None),
                    second_per_grid_ts=inputs.get("second_per_grid_ts", None),
                    max_new_tokens=eval_args.max_new_tokens,
                    do_sample=eval_args.do_sample,
                    temperature=eval_args.temperature,
                    top_p=eval_args.top_p,
                    top_k=eval_args.top_k,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            
            # Decode generated text
            generated_text = processor.tokenizer.decode(
                generated_ids[0][prompt_length:], 
                skip_special_tokens=True
            )
            
            # Decode ground truth if available
            if labels is not None:
                response_labels = labels[labels != -100]
                ground_truth_text = processor.tokenizer.decode(
                    response_labels, 
                    skip_special_tokens=True
                )
            else:
                ground_truth_text = ""
            
            # Store full example info
            example_info = {
                "index": i,
                "prediction": generated_text,
                "ground_truth": ground_truth_text,
            }
            
            # Add original data if available
            if hasattr(eval_dataset, "list_data_dict") and i < len(eval_dataset.list_data_dict):
                example_info["original_data"] = eval_dataset.list_data_dict[i]
            
            predictions.append(example_info)

    # Save results
    output_file = eval_args.output_file or os.path.join(eval_args.output_dir, "eval_results.json")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    results = {
        "metrics": metrics,
        "num_examples": len(eval_dataset),
    }
    
    if eval_args.save_predictions:
        results["predictions"] = predictions
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    rank0_print(f"Evaluation results saved to {output_file}")
    
    # Print summary
    rank0_print("\n" + "="*50)
    rank0_print("Evaluation Summary")
    rank0_print("="*50)
    for key, value in metrics.items():
        rank0_print(f"{key}: {value}")
    rank0_print("="*50)


if __name__ == "__main__":
    evaluate()

