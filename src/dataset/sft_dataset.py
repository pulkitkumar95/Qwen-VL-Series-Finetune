import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai, pad_sequence

def point_sampler(obj_ids, num_pt_points_per_obj, sampling_method='random'):
    all_obj_ids = obj_ids.cpu().numpy()
    sampled_pt_indices = []
    if sampling_method == 'random':
        for single_unique_obj_id in np.unique(all_obj_ids):
            obj_id_indices = np.where(all_obj_ids == single_unique_obj_id)[0]
            num_pts_for_obj = min(num_pt_points_per_obj, len(obj_id_indices))
            sampled_indices = np.random.choice(obj_id_indices, num_pts_for_obj, replace=False)
            sampled_pt_indices.extend(sampled_indices)
    else:
        raise ValueError(f"Sampling method not supported: {sampling_method}")
    
    return sampled_pt_indices
   
def preprocess_pt_data(pt_data, video_metadata, num_pt_points_per_obj,
                       pt_sampling_method='random', temporal_patch_size=2):
    max_y, max_x = video_metadata['height'], video_metadata['width']
    pred_tracks = pt_data['pred_tracks']
    pred_visibility = pt_data['pred_visibility']
    frames_used_for_pt = pred_tracks.shape[0]
    frames_selected_for_video = video_metadata['frames_indices']
    num_frames_selected = len(frames_selected_for_video)
    pt_frame_indices_to_use = np.linspace(0, frames_used_for_pt - 1, num_frames_selected).astype(int)
    pred_visibility = pred_visibility[pt_frame_indices_to_use] 
    pred_tracks = pred_tracks[pt_frame_indices_to_use]
    obj_ids = pt_data['obj_ids']
    #normalize to -1 to 1
    div_factor = torch.tensor([max_x, max_y]).view(1, 1, 2)
    pred_tracks = pred_tracks / div_factor
    pred_tracks = (pred_tracks - 0.5)/ 0.5
    sampled_pt_indices = point_sampler(obj_ids, num_pt_points_per_obj, pt_sampling_method)
    pred_tracks = pred_tracks[:, sampled_pt_indices]
    pred_visibility = pred_visibility[:,sampled_pt_indices]
    pt_data_to_return = {
        'pred_tracks': pred_tracks[::temporal_patch_size],
        'pred_visibility': pred_visibility[::temporal_patch_size],
        'obj_ids': pt_data['obj_ids'][sampled_pt_indices],
    }
    return pt_data_to_return

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
        mode='train'
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
            self.return_video_metadata = True
        else:
            self.image_patch_size = 14
            self.return_video_metadata = False

        self.mode = mode
        self.use_pt = data_args.use_pt
        self.pt_folder = data_args.pt_folder
        self.pt_name = data_args.pt_name
        self.pt_dataset_name = data_args.pt_dataset_name
        self.num_pt_points_per_obj = data_args.num_pt_points_per_obj
        self.pt_sampling_method = data_args.pt_sampling_method

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        og_data = sources.copy()
        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []

            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                image_input = get_image_info(
                        image_file, 
                        self.image_min_pixel, 
                        self.image_max_pixel, 
                        self.image_resized_w, 
                        self.image_resized_h, 
                        self.image_patch_size
                    )
                images.append(image_input)

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]
                if 'duration' in sources:
                    duration = float(sources['duration'])
                else:
                    duration = None

            videos = []
            for video_file in video_files:
                og_video_file = video_file
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(
                    video_file, 
                    self.video_min_pixel, 
                    self.video_max_pixel, 
                    self.video_resized_w, 
                    self.video_resized_h, 
                    self.data_args.fps,
                    self.image_patch_size,
                    return_video_metadata=self.return_video_metadata,
                    nframes=self.data_args.nframes,
                    duration=duration
                )
                if isinstance(video_input, tuple):
                    video_metadata = video_input[1]
                if self.use_pt:
                    pt_file = og_video_file.replace(".mp4", ".pkl")
                    pt_path = os.path.join(self.pt_folder, self.pt_name, self.pt_dataset_name, 'feat_dump', pt_file)
                    pt_data = pickle.load(open(pt_path, "rb"))
                    pt_data = preprocess_pt_data(pt_data, video_metadata, self.num_pt_points_per_obj, self.pt_sampling_method)
                else:
                    pt_data = None
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        image_curr_count = 0
        video_curr_count = 0
        
        # Qwen2-VL uses a default system message so I've added this.
        # Qwen3-Vl does not use a system message by default.
        if len(SYSTEM_MESSAGE) > 0 and "Qwen3" not in self.model_id:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input:
                num_images = user_input.count(DEFAULT_IMAGE_TOKEN)
                # Slice the images list to get the images for the current turn.
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(text=[user_input], images=images_for_this_turn, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images

            elif DEFAULT_VIDEO_TOKEN in user_input:
                num_videos = user_input.count(DEFAULT_VIDEO_TOKEN)
                # Slice the videos list to get the videos for the current turn.
                videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                if "Qwen2.5" in self.model_id:
                    inputs = processor(
                        text=[user_input], 
                        images=images, 
                        videos=videos_for_this_turn, 
                        padding=False, 
                        do_resize=False, 
                        return_tensors='pt', 
                        **video_kwargs
                    )
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                elif "Qwen3" in self.model_id:

                    videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                    video_datas_for_turn, video_metadatas_for_turn = zip(*videos_for_this_turn)
                    video_datas_for_turn = list(video_datas_for_turn)
                    video_metadatas_for_turn = list(video_metadatas_for_turn)

                    inputs = processor(
                        text=[user_input],
                        images=images,
                        videos=video_datas_for_turn,
                        padding=False,
                        do_resize=False,
                        return_tensors='pt',
                        **video_kwargs,
                        video_metadata=video_metadatas_for_turn,
                        pt_data=pt_data,
                    )
                else:
                    inputs = processor(
                        text=[user_input], 
                        images=images, 
                        videos=videos_for_this_turn, 
                        padding=False, 
                        do_resize=False, 
                        return_tensors='pt'
                    )
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                video_curr_count += num_videos

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
            prompt_len = len(prompt_input_ids[0])
            if self.mode=='train':
                input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
                # In causal LMs, logits[i] predicts input_ids[i+1]
                # So labels[i] should be input_ids[i+1]
                # For response: we want to predict response_token_0 from the last prompt position
                # response_labels = response_input_ids.squeeze(0)
                
                # # Create labels: [IGNORE for prompt, response tokens shifted, IGNORE at end]
                # # labels[i] = input_ids[i+1], so labels[prompt_len-1] = response_token_0
                # labels = torch.cat(
                #     [
                #         torch.tensor([IGNORE_INDEX] * (prompt_len - 1)),  # All prompt tokens except last
                #         response_labels,  # Response tokens: will be at positions [prompt_len-1, prompt_len, ...]
                #         torch.tensor([IGNORE_INDEX]),  # No token to predict after last response token
                #     ],
                #     dim=0,
                # )

                labels = torch.cat(
                    [
                        torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                        response_input_ids.squeeze(0),
                    ],
                    dim=0,
                )
            else:
                input_ids = prompt_input_ids.squeeze(0)
                labels = response_input_ids.squeeze(0)

        

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if self.mode=='test':
            data_dict['question_key'] = og_data['question_key']
            data_dict['task_type'] = og_data['task_type']
        del og_data
        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird

        if self.use_pt:
            data_dict["pred_tracks"] = pt_data["pred_tracks"]
            data_dict["pred_visibility"] = pt_data["pred_visibility"]
            data_dict["obj_ids"] = pt_data["obj_ids"]

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, padding_side='left'):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        question_ids = []
        task_types = []
 
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            if "question_key" in keys:
                question_ids.append(example["question_key"])
            if "task_type" in keys:
                task_types.append(example["task_type"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side=self.padding_side, padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side=self.padding_side, padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts
        
        if len(question_ids) > 0:
            data_dict["question_key"] = question_ids
        if len(task_types) > 0:
            data_dict["task_type"] = task_types

        return data_dict

def make_supervised_data_module(model_id, processor, data_args, mode='train'):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id,
        mode=mode
    )
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = SupervisedDataset(
              data_path=data_args.eval_path,
              processor=processor,
              data_args=data_args,
              model_id=model_id,
              mode=mode
          )
        
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id,
    padding_side=processor.tokenizer.padding_side)

    return dict[str, SupervisedDataset | DataCollatorForSupervisedDataset | None](train_dataset=sft_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
