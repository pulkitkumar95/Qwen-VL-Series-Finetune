import os
import torch
import random
import ast
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
from src.trainer import QwenSFTTrainer
from src.dataset import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from monkey_patch_forward import (
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward, 
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward
)
from monkey_patch_vision import replace_qwen2_5_vision

import numpy as np
from src.constants import IGNORE_INDEX
import json
local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def eval_favor(eval_results):
    all_dict = {
        "ALL":[0,8184],
        "AS":[0,2637],
        "HAC":[0,1541],
        "SAD":[0,1662],
        "MAD":[0,1205],
        "CM":[0,1075],
        "NSM":[0,64]
    }
    for key, result_dict in eval_results.items():
        task_type = result_dict["task_type"]
        if result_dict["judge"] == True:
            all_dict[task_type][0] += 1
            all_dict["ALL"][0] += 1

    scores1 = {key: round(value[0] / value[1] * 100, 2) for key, value in all_dict.items()}
    scores = [round(value[0] / value[1] * 100, 2) for value in all_dict.values()]
    formatted_output = " & ".join([f"{score}" for score in scores])
    print(formatted_output)
    return scores1, formatted_output

def og_favor_per_question_eval(output_text, correct_answer, options):
    "taken from https://github.com/FAVOR-Bench/FAVOR-Bench/blob/main/inference_qa_qwen.py"

    containing_options = [opt for opt in options if opt != correct_answer and correct_answer in opt]

    if not containing_options:
        if correct_answer.lower() in output_text.lower():
            judge = True
        else:
            judge = False
    else:
        if correct_answer.lower() in output_text.lower():
            judge = True
            for option in containing_options:
                if option.lower() in output_text.lower():
                    judge = False
        else:
            judge = False
    return judge


def build_eval_metrics_fn(processor, eval_path):
    tokenizer = processor.tokenizer

    def _first_letter(text: str):
        for ch in text.strip():
            if ch.isalpha():
                return ch.upper()
        return None
    judge_results = []
    target_text_list = []
    pred_text_list = []
    out_save_dict = {}
    main_question_list = json.load(open(eval_path, "r"))
    main_question_dict = {item["question_key"]: item for item in main_question_list}

    def compute_metrics(eval_prediction, compute_result=False):
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
        question_keys = eval_prediction.inputs['question_key']
        task_types = eval_prediction.inputs['task_type']

        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if isinstance(labels, (tuple, list)):
            labels = labels[0]
        

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        
        for target_ids, pred_ids, question_key, task_type in zip(labels, logits, question_keys, task_types):
            target_ids = target_ids[target_ids!=IGNORE_INDEX]
            pred_ids = pred_ids[pred_ids!=IGNORE_INDEX]
            target_tokens = target_ids.tolist()
            pred_tokens = pred_ids.tolist()
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            actual_options = main_question_dict[question_key]["options"]
            eval_type = main_question_dict[question_key]["eval_type"]
            if eval_type == 'option_letter':
                target_letter = _first_letter(target_text)
                pred_letter = _first_letter(pred_text)
                judge_results.append(target_letter == pred_letter)
                judge = target_letter == pred_letter
            elif eval_type == 'full_text':
                judge = og_favor_per_question_eval(target_text, pred_text, actual_options)
                judge_results.append(judge)

            
            target_text_list.append(target_text)
            pred_text_list.append(pred_text)
            out_save_dict[question_key] = {
                "target_text": target_text,
                "pred_text": pred_text,
                "task_type": task_type,
                "judge": judge,
            }


        judge_accuracy = (
            float(np.mean(judge_results)) if judge_results else 0.0
        )
        if compute_result:
            return {
                'all_results': out_save_dict,
                'judge_accuracy': judge_accuracy,
            }


        return {
            # "token_accuracy": token_accuracy,
            "accuracy": judge_accuracy,
        }

    return compute_metrics

def is_lora_checkpoint(checkpoint_path):
    """Check if checkpoint contains LoRA adapter files."""
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    return os.path.exists(adapter_config_path)

def load_model_from_checkpoint(checkpoint_path, model_args, training_args):
    """Load model from checkpoint, handling both LoRA and full model checkpoints."""
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))
    
    # Check if this is a LoRA checkpoint
    if is_lora_checkpoint(checkpoint_path):
        rank0_print(f"Loading LoRA checkpoint from {checkpoint_path}")
        
        # Load base model first
        base_model_id = model_args.model_id
        if "Qwen2.5" in base_model_id:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )
        else:
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )
        
        # Load non-LoRA weights if they exist
        non_lora_path = os.path.join(checkpoint_path, "non_lora_state_dict.bin")
        if os.path.exists(non_lora_path):
            rank0_print("Loading non-LoRA weights...")
            non_lora_state_dict = torch.load(non_lora_path, map_location="cpu")
            # Handle key name variations
            non_lora_state_dict = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_state_dict.items()}
            if any(k.startswith('model.model.') for k in non_lora_state_dict):
                non_lora_state_dict = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_state_dict.items()}
            base_model.load_state_dict(non_lora_state_dict, strict=False)
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        rank0_print("LoRA checkpoint loaded successfully")
    else:
        # Load full model checkpoint
        rank0_print(f"Loading full model checkpoint from {checkpoint_path}")
        
        # Check model architecture from config
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            is_qwen2_5 = "Qwen2_5" in config.get("architectures", [""])[0] if config.get("architectures") else False
            is_qwen3 = "Qwen3" in config.get("architectures", [""])[0] if config.get("architectures") else False
        else:
            # Fallback to model_id
            is_qwen2_5 = "Qwen2.5" in model_args.model_id
            is_qwen3 = "Qwen3" in model_args.model_id
        
        if is_qwen3:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )
        elif is_qwen2_5:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )
        rank0_print("Full model checkpoint loaded successfully")
    
    model.config.use_cache = True
    model.eval()
    
    return model

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: torch.use_deterministic_algorithms(True) can be set but may impact performance
    # Uncomment the line below if you need full determinism (may be slower)
    # torch.use_deterministic_algorithms(True, warn_only=True)

def evaluate():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed for reproducibility
    seed = training_args.seed if hasattr(training_args, 'seed') and training_args.seed is not None else 42
    set_seed(seed)
    rank0_print(f"Random seed set to {seed} for reproducibility")
    
    # Ensure eval_path is provided
    if data_args.eval_path is None:
        raise ValueError("--eval_path must be provided for evaluation. Please specify the path to your evaluation JSON file.")
    
    
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")
    
    local_rank = training_args.local_rank
    
    # Load model from checkpoint
    # If output_dir is specified and contains a checkpoint, use it; otherwise use model_id
    checkpoint_path = None
    if hasattr(training_args, 'output_dir') and training_args.output_dir:
        # Check for checkpoints in output_dir
        checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        if checkpoint_dirs:
            # Use the latest checkpoint
            checkpoint_path = str(sorted(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]))[-1])
            rank0_print(f"Found checkpoint: {checkpoint_path}")
        elif os.path.exists(training_args.output_dir) and (os.path.exists(os.path.join(training_args.output_dir, "config.json")) or is_lora_checkpoint(training_args.output_dir)):
            # output_dir itself is a checkpoint
            checkpoint_path = training_args.output_dir
            rank0_print(f"Using output_dir as checkpoint: {checkpoint_path}")
    
    if checkpoint_path:
        model = load_model_from_checkpoint(checkpoint_path, model_args, training_args)
        # Load processor from checkpoint if available, otherwise from model_id
        processor_path = checkpoint_path if os.path.exists(os.path.join(checkpoint_path, "tokenizer_config.json")) else model_args.model_id
        processor = AutoProcessor.from_pretrained(processor_path)
        processor.tokenizer.padding_side = 'left'
    else:
        # Load from base model_id
        rank0_print(f"No checkpoint found, loading base model from {model_args.model_id}")
        compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        
        bnb_model_from_pretrained_args = {}
        if training_args.bits in [4, 8]:
            bnb_model_from_pretrained_args.update(dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["visual", "lm_head"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                )
            ))
        
        config = AutoConfig.from_pretrained(model_args.model_id)

        if config.model_type == "qwen3_vl_moe":
            replace_qwen3_vl_moe_with_mixed_modality_forward()
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )

        elif config.model_type == "qwen3_vl":
            replace_qwen3_with_mixed_modality_forward()
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )

        elif config.model_type == "qwen2_5_vl":
            replace_qwen2_5_with_mixed_modality_forward()
            replace_qwen2_5_vision()
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
                **bnb_model_from_pretrained_args
            )
            
        else:
            replace_qwen_2_with_mixed_modality_forward()
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_args.model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
                **bnb_model_from_pretrained_args
            )
        
        model.config.use_cache = True
        model.eval()
        processor = AutoProcessor.from_pretrained(model_args.model_id)
        processor.tokenizer.padding_side = 'left'
    
    # Create evaluation dataset
    # Temporarily set data_path to eval_path for dataset creation
    # We'll only use eval_dataset from the data_module
    original_data_path = data_args.data_path
    original_image_folder = data_args.image_folder
    
    data_args.data_path = data_args.eval_path  # Use eval_path for dataset initialization
    # Use eval_image_folder if provided, otherwise use image_folder
    if data_args.eval_image_folder is not None:
        data_args.image_folder = data_args.eval_image_folder
    
    data_module = make_supervised_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
        mode='test'
    )
    
    # Restore original values
    data_args.data_path = original_data_path
    data_args.image_folder = original_image_folder
    
    if data_module["eval_dataset"] is None:
        raise ValueError("Failed to create evaluation dataset. Please check your --eval_path and --eval_image_folder arguments.")
    
    compute_metrics = build_eval_metrics_fn(processor, data_args.eval_path)

    # Create trainer for evaluation
    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
        compute_metrics=compute_metrics,
    )
    
    # Run evaluation
    rank0_print("Starting evaluation...")
    eval_results = trainer.evaluate(log_metrics=False)
    # Gather per-rank result dicts (if present) so rank 0 can see everything.
    if 'eval_all_results' in eval_results and torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        gathered_results = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_results, eval_results['eval_all_results'])
        merged_results = {}
        for res in gathered_results:
            if res:
                merged_results.update(res)
        eval_results['eval_all_results'] = merged_results
        fist_letter_accuracy = []
        for key, value in merged_results.items():
            fist_letter_accuracy.append(value['judge'])
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        eval_results['first_letter_accuracy'] = np.mean(fist_letter_accuracy)
        if data_args.eval_dataset_name == "favor":
            scores1, _ = eval_favor(eval_results['eval_all_results'])
            for key, score in scores1.items():
                eval_results[f"{data_args.eval_dataset_name}/{key}"] = score

        if data_args.result_dump_dir is not None:
            with open(os.path.join(data_args.result_dump_dir, 'eval_results.json'), 'w') as f:
                json.dump(eval_results['eval_all_results'], f, indent=2, ensure_ascii=False)
    
    rank0_print("\n" + "="*50)
    rank0_print("Evaluation Results:")
    rank0_print("="*50)
    if 'eval_all_results' in eval_results:
        rank0_print(len(eval_results['eval_all_results']))
        del eval_results['eval_all_results']
    for key, value in eval_results.items():
        rank0_print(f"{key}: {value}")
    trainer.log(eval_results)
    rank0_print("="*50)
    
    return eval_results


if __name__ == "__main__":
    evaluate()

