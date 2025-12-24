"""Slurm generator"""
import os
import argparse
import time
import sys
import getpass

from collections import defaultdict
import socket
import numpy as np
from slurm_funcs import get_slurm_init_commands, args_specific_commands, main_slurm_run_commands


#pylint: disable=redefined-outer-name,unspecified-encoding,invalid-name
def get_args():
    """Get arparse stuff"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--nhrs', type=int, default=24)
    parser.add_argument('--base-dir', default=f'{os.getcwd()}')
    parser.add_argument('--output-dirname', default='out_new_pt')
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--exp_name', required=True, type=str, help='Experiment name')
    parser.add_argument('--qos', default="vulc_scav", type=str, help='Qos to run')
    parser.add_argument('--cores', default=16, type=int, help='Number of cpu cores')
    parser.add_argument('--mem', default=120, type=int, help='RAM in G')
    parser.add_argument('--gpu', default=1, type=int, help='Number of gpus')
    parser.add_argument('--gpus_to_use', default='h200', type=str, help='GPUs to use')
    parser.add_argument('--batch_size_per_device', default=4, type=int, help='Batch size per device')
    parser.add_argument('--per_device_eval_batch_size', default=2, type=int, help='Batch size per device for evaluation')
    parser.add_argument('--run_by_gh200', action='store_true', help='Run by gh200')
    parser.add_argument('--mode', default='train', type=str, help='Mode to run')
    parser.add_argument('--eval_path_dir', default=None, type=str, help='Evaluation path')

    return parser.parse_args()


def generate_commands(args):
    """Generate commands"""
    singulairty_cmd = ''
    hostname = socket.gethostname()
    if 'zara' in hostname:
        docker_command = 'singularity'
        username = getpass.getuser()
        bind_paths = f'/scratch/zt1/project/abhinav2-prj/user/{username}'
        image_path = '/scratch/zt1/project/abhinav2-prj/user/pulkit/orvit_pt/cotracker.sif'
        singulairty_cmd = f'{docker_command} exec --bind {bind_paths} --nv {image_path} '
 
   

    time_str = str(int(time.time()))



    params = []
    command_file = None
    port_start = 1000 + (int(np.random.uniform()*10e5) % 64000)
    slurm_stuff = defaultdict(list)
    slurm_stuff_dir =  os.path.join(os.getcwd(), 'out_slurm', args.mode, args.exp_name)
    os.makedirs(slurm_stuff_dir, exist_ok=True)
    vlm_model_dict = {
        'qwen3vl4b': 'Qwen/Qwen3-VL-4B-Instruct',
        'qwen25vl7b': 'Qwen/Qwen2.5-VL-7B-Instruct',
    }
    data_path_dict = {
        'llava_9k': {
            'data_json_path': '/fs/vulcan-projects/motion_llm/pulkit/Flash-VStream/Flash-VStream-Qwen/data/llava-video-178k/trainset_9k_video.json',
            'image_folder_path': '/fs/cfar-projects/actionloc/data/llava_video_9k',
        },
        'motionbench': {
            'data_json_path': '/fs/vulcan-projects/motion_llm/pulkit/dataset_utils/final_jsons/motionbench_llava_style_option_letter.json',
            'image_folder_path': '/fs/vulcan-projects/motion_llm/datasets/MotionBench/MotionBench/all_vids',
        },
        'favor': {
            'data_json_path': '/fs/vulcan-projects/motion_llm/pulkit/dataset_utils/final_jsons/favor_llava_style_option_letter.json',
            'image_folder_path': '/fs/cfar-projects/actionloc/hub/datasets--zl2048--FAVOR/snapshots/2a78953831e41ebd046a7e6e55eb3b6c28f61e9b/videos/FAVOR-Bench',
        },
        'vlm4d': {
            'data_json_path': '/fs/vulcan-projects/motion_llm/pulkit/dataset_utils/final_jsons/vlm4d_llava_style_option_letter.json',
            'image_folder_path': '/fs/vulcan-projects/motion_llm/datasets/VLM4D/',
        },
        'wolf': {
            'data_json_path': '/fs/vulcan-projects/motion_llm/pulkit/dataset_utils/final_jsons/wolf_llava_style_option_letter.json',
            'image_folder_path': '/fs/vulcan-projects/motion_llm/datasets/WoWolf_v2-dev',
        },
    }

    if args.mode == 'train':
        datasets_to_use = ['llava_9k']
    elif args.mode == 'eval':
        datasets_to_use = ['motionbench']
        assert args.eval_path_dir is not None, 'Evaluation path is required'
        print(f'Evaluating on: {args.eval_path_dir}')

    params = [(base_model, dataset_to_use) for base_model in ['qwen3vl4b']
                                            for dataset_to_use in datasets_to_use]
    DEFAULT_TRAIN_EFFECTIVE_BATCH_SIZE = 128
    GRAD_ACCUM_STEPS = DEFAULT_TRAIN_EFFECTIVE_BATCH_SIZE // (args.batch_size_per_device * args.gpu)
    VIDEO_MAX_PIXELS = 360 * 420

    for i, (base_model, dataset_to_use) in enumerate(params):

        log_path = f'logs/{time_str}/{i}.log'
        log_dir = os.path.dirname(log_path) 
        add_str = f'{base_model}_{dataset_to_use}'
        cmd = singulairty_cmd
        output_dir = os.path.join(slurm_stuff_dir, add_str)
        os.makedirs(output_dir, exist_ok=True)
        log_dir = os.path.join(output_dir, 'log.txt')
        err_dir = os.path.join(output_dir, 'err.txt')
        slurm_stuff['log'].append(log_dir)
        slurm_stuff['err'].append(err_dir)
        tabspace = ''
        master_port = port_start + i
        # Build command
        cmd += f'deepspeed --master_port {master_port} src/train/{args.mode}_sft.py {tabspace}'
        cmd += f' --use_liger_kernel False {tabspace}'
        cmd += f' --deepspeed scripts/zero3.json {tabspace}'
        cmd += f' --remove_unused_columns False {tabspace}'
        cmd += f' --freeze_vision_tower True {tabspace}'
        cmd += f' --freeze_llm False {tabspace}'
        cmd += f' --freeze_merger False {tabspace}'
        cmd += f' --bf16 True {tabspace}'
        cmd += f' --fp16 False {tabspace}'
        cmd += f' --disable_flash_attn2 False {tabspace}'
        cmd += f' --num_train_epochs 1 {tabspace}'
        cmd += f' --per_device_train_batch_size {args.batch_size_per_device} {tabspace}'
        cmd += f' --gradient_accumulation_steps {GRAD_ACCUM_STEPS} {tabspace}'
        cmd += f' --video_max_pixels {VIDEO_MAX_PIXELS} {tabspace}'
        cmd += f' --fps 1 {tabspace}'
        cmd += f' --learning_rate 1e-5 {tabspace}'
        cmd += f' --merger_lr 1e-5 {tabspace}'
        cmd += f' --vision_lr 2e-6 {tabspace}'
        cmd += f' --weight_decay 0.1 {tabspace}'
        cmd += f' --warmup_ratio 0.03 {tabspace}'
        cmd += f' --lr_scheduler_type cosine {tabspace}'
        cmd += f' --logging_steps 1 {tabspace}'
        cmd += f' --tf32 True {tabspace}'
        cmd += f' --gradient_checkpointing True {tabspace}'
        cmd += f' --report_to wandb {tabspace}'
        cmd += f' --lazy_preprocess True {tabspace}'
        cmd += f' --save_strategy steps {tabspace}'
        cmd += f' --save_steps 1000 {tabspace}'
        cmd += f' --save_total_limit 1 {tabspace}'
        cmd += f' --dataloader_num_workers {args.cores} {tabspace}'
        cmd += f' --model_id {vlm_model_dict[base_model]} {tabspace}'
        cmd += f' --image_folder {data_path_dict[dataset_to_use]["image_folder_path"]} {tabspace}'
        if args.mode == 'train':
            cmd += f' --output_dir {output_dir} {tabspace}'
            cmd += f' --data_path {data_path_dict[dataset_to_use]["data_json_path"]} {tabspace}'
        elif args.mode == 'eval':
            cmd += f' --eval_path {data_path_dict[dataset_to_use]["data_json_path"]} {tabspace}'
            cmd += f' --output_dir {args.eval_path_dir} {tabspace}'
            cmd += f' --eval_dataset_name {dataset_to_use} {tabspace}'
            cmd += f' --per_device_eval_batch_size {args.per_device_eval_batch_size} {tabspace}'
            cmd += f' --batch_eval_metrics True {tabspace}'
            cmd += f' --include_for_metrics inputs {tabspace}'
            cmd += f' --result_dump_dir {output_dir} {tabspace}'
        slurm_stuff['commands'].append(cmd)
    
    for key, value in slurm_stuff.items():
        with open(os.path.join(slurm_stuff_dir, f'{key}.txt'), 'w') as f:
            for item in value:
                f.write(f'{item}\n')
    destination_dir = ''
    return slurm_stuff_dir, len(slurm_stuff['commands']), destination_dir


if __name__ == '__main__':
    args = get_args()
    hostname = socket.gethostname()
    if 'zara' in hostname:
        zara = True
        args.gpu = 1
        args.cores = 32
        args.mem = 120
    else:
        zara = False

    slurm_stuff_dir, num_commands, data_dir = generate_commands(args)

    slurm_file_path = os.path.join(slurm_stuff_dir, f'{args.exp_name}.sbatch')
    slurm_file_stuff = []
    remove_nodes = ['clip', 'gamma', 'vulcan42', 'cbcb']
    slurm_file_stuff += get_slurm_init_commands(args, num_commands)
    slurm_file_stuff += args_specific_commands(args, remove_nodes, zara)
    slurm_file_stuff += main_slurm_run_commands(args, slurm_stuff_dir, data_dir, zara)

    with open(slurm_file_path, 'w') as f:
        for line in slurm_file_stuff:
            f.write(line)
    print(f'sbatch {slurm_file_path}')

    if not args.dryrun and not args.run_by_gh200:
        os.system(f'sbatch {slurm_file_path}')
