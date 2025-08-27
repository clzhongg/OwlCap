# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
from pathlib import Path
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import requests
import time 
import sys
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
# from transformers import Qwen2VLForConditionalGeneration
project_root = str(Path(__file__).resolve().parent.parent.parent)  # 向上三级到 VideoChat-R1-main
sys.path.append(project_root)
from math_verify import parse, verify
from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video_Caption_nothink as Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from src.open_r1.my_qwen_utils import process_vision_info
from tqdm import tqdm
import torch
import json
import random
import ast
import re
from glob import glob
from Levenshtein import distance
import pathlib


def count_english_and_chinese(sentence):
    # 匹配英文单词
    english_words = re.findall(r'\b[a-zA-Z]+\b', sentence)
    english_count = len(english_words)
    
    # 匹配中文字符
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', sentence)
    chinese_count = len(chinese_characters)
    
    return english_count, chinese_count
def count_length(sentence):
    # 匹配英文单词（含连字符、缩写、所有格等）
    english_words = re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", sentence)
    return len(english_words)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "answer"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="/share/wy/Video/Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    # preprocessed_data_path: Optional[str] = field( # Add preprocessed_data_path argument
    #     default="",
    #     metadata={"help": "Path to the preprocessed dataset directory. If provided, load preprocessed data instead of raw videos."},
    # )


def is_valid_two_d_list_format(s):
    pattern = r'^\[(\(\d+(\.\d+)?,\s*\d+(\.\d+)?\)(,\s*\(\d+(\.\d+)?,\s*\d+(\.\d+)?\))*(,)?|)\]$'
    if not re.match(pattern, s):
        return False
    try:
        # 尝试将字符串转换为 Python 对象
        lst = ast.literal_eval(s)
        # 检查对象是否为列表
        if not isinstance(lst, list):
            return False
        # 检查列表中的每个元素是否为元组
        for item in lst:
            if not isinstance(item, tuple):
                return False
            # 检查元组是否包含两个元素
            if len(item) != 2:
                return False
            # 检查元组中的元素是否为数字
            for num in item:
                if not isinstance(num, (int, float)):
                    return False
            if item[0] > item[1]: # 保证符合时序区间
                return False
        return True
    except:
        return False
        

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r'\s*<answer>.*?</answer>\s*<glue>.*?</glue>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    # print('matches:', matches)
    reward_list = []
    for i, match in enumerate(matches):
        if match:
            pattern_glue = r'<glue>(.*?)</glue>'

            # 使用 search 方法查找首个匹配项
            match_glue = re.search(pattern_glue, completions[i], re.DOTALL)

            if match_glue:
                # 获取捕获组中的内容
                glue = match_glue.group(1)
            else:
                raise ValueError(completions[i])

            if is_valid_two_d_list_format(glue):
                r = 1.0
            else:
                r = 0.0
        else:
            r = 0.0
        reward_list.append(r)
    return reward_list

def check_first_frame(pred):
    if '第一帧' in pred or 'in one frame' in pred or 'in one of the frames' in pred or 'first Frame' in pred or 'second Frame' in pred or "initial Frame" in pred or 'the first scene' in pred:
        return True
    else:
        return False
        

def length_reward(completions, solution, **kwargs):
    reward_list=[]
    for content, sol in zip(completions, solution): 
        sol=sol['answer']
        sline_count=content.count('\n')

        end_string='。？！….?!'
        if content[-1] in end_string:
            end_flag=True
        else:
            end_flag=False

        frame_flag=check_first_frame(content)
        #######################################  先判断语种
        en_cnt=count_length(content)
        len_content=en_cnt
        if (len_content>=100 and len_content<=300): # or (len_content>=len_sol-50 and len_content<=len_sol+50)
            reward_list.append(1)
        elif (len_content>=300 and len_content<=400):
            reward_list.append(0.5)
        else:
            reward_list.append(0.2)

    return reward_list
import json

def extract_and_analyze_json(text):
    """
    从文本中提取JSON数据并分析Pred_units和Gt_units中response为'Yes'的比例
    
    参数:
    text (str): 包含JSON数据的文本
    
    返回:
    tuple: 包含Pred_units和Gt_units中response为'Yes'的单元比例
    """
    try:
        # 定位第一个</think>标签后的JSON数据
        think_end = text.find('</think>')
        if think_end == -1:
            return 0.0, 0.0
        
        json_text = text[think_end + len('</think>'):].strip()
        
        # 提取JSON对象（从第一个{到最后一个}）
        start_idx = json_text.find('{')
        end_idx = json_text.rfind('}') + 1
        if start_idx == -1 or end_idx <= start_idx:
            return 0.0, 0.0
        
        json_str = json_text[start_idx:end_idx]
        
        # 解析JSON
        data = json.loads(json_str)
        
        # 安全地获取Pred_units和Gt_units
        pred_units = data.get('Pred_units', [])
        gt_units = data.get('Gt_units', [])
        
        # 计算Pred_units中response为'Yes'的比例
        pred_yes = 0
        pred_total = len(pred_units)
        if pred_total > 0:
            for item in pred_units:
                if isinstance(item, dict) and item.get('response') == 'Yes':
                    pred_yes += 1
        pred_ratio = pred_yes / pred_total if pred_total > 0 else 0.0
        
        # 计算Gt_units中response为'Yes'的比例
        gt_yes = 0
        gt_total = len(gt_units)
        if gt_total > 0:
            for item in gt_units:
                if isinstance(item, dict) and item.get('response') == 'Yes':
                    gt_yes += 1
        gt_ratio = gt_yes / gt_total if gt_total > 0 else 0.0
        
        return round(pred_ratio, 2), round(gt_ratio, 2)
    
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        # 捕获JSON解析错误、类型错误或键错误等
        return 0.0, 0.0
import logging
import time
import json
import requests

def correctness_reward_vllm(completions, solution, **kwargs):
    # 配置日志
    
    reward_list = []
    
    # vllm的API地址
    url = 'http://127.0.0.1:8000/v1/chat/completions'
    headers = {'Content-Type': 'application/json'}
    
    for content, sol in zip(completions, solution): 
        sol = sol['answer']
        prompt=f'''
    You are now a professional language analysis expert and need to analyze the degree of overlap between two captions (pred_caption and gt_caption) following these specific steps:
    First: Splitting of original information units: Split pred_caption and gt_caption into original information units respectively. An original information unit should be a simple statement. The facts it contains must be the smallest information fragments that would lose their meaning if split further. If possible, abstract concepts or broad interpretations should be simplified into more basic, constitutive observations. An original information unit should contain only one main element.
    Second: Matching of original information units: Match each original information unit of the split pred_caption with gt_caption one by one, and determine whether the content of the original information unit is consistent with that of gt_caption. The judgment of consistency is based on whether the content of the original information unit conflicts with gt_caption. Different descriptions of the same information are also considered correct. You need to judge whether the original information unit conforms to the description based on your overall understanding of the captions.
    Similarly, match each original information unit of the split gt_caption with pred_caption one by one, and determine whether the content of the original information unit is consistent with that of pred_caption. The matching results are indicated by "Yes" (consistent) or "No" (inconsistent).
    Third: Output format: Output the results strictly in the following JSON format:
    {{
    "Pred_units": [{{"units": "First Unit", "response": "Yes"}}, {{"units": "Second unit", "response": "No"}}, ...],
    "Gt_units": [{{"units": "First Unit", "response": "Yes"}}, {{"units": "Second unit", "response": "No"}}, ...]
    }}   
    This is the pred_caption:{content}.
    This is the gt_caption: {sol}
        '''
        request_data = {
            "model": "/home/notebook/data/group/group/Zhongchunlin/VideoChat-R1-main/Qwen3-32B",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 8192,
            "extra_body":{
                "top_k": 20, 
                "chat_template_kwargs": {"enable_thinking": False},
            }
        }
        start_time = time.time()
        response = requests.post(url, data=json.dumps(request_data), headers=headers)
        response.raise_for_status()
        
        response_data = response.json()
        generated_text = response_data['choices'][0]['message']['content']
        end_time = time.time()
        elapsed_time = end_time - start_time
        correctness, completeness = extract_and_analyze_json(generated_text)
        
        # 计算奖励
        reward = correctness + completeness
        reward_list.append(reward)
        # print(reward)
    
    return reward_list
    




reward_funcs_registry = {
    "format": length_reward,
    "correctness":correctness_reward_vllm
}

def read_jsonldataset(data_path, repeat_time, random_seed=2025):
    raw_data = []
    filepaths = sorted(glob(data_path))

    for temp_filename in filepaths:
        with open(temp_filename, 'r') as f:
            lines = f.readlines()
            if repeat_time < 1:
                num_lines = int(len(lines) * repeat_time)
                used_lines = lines[:num_lines]
            else:
                used_lines = lines * int(repeat_time)
                # num_lines = int(len(lines))
            # used_lines = lines[:num_lines]
            raw_data.extend(used_lines)

    return raw_data


def load_json_dataset(train_data_path, eval_data_path, video_folder):#, preprocessed_data_path=None): # Modified to accept preprocessed_data_path
    def create_dataset_from_json(file_lists, split_name):
        examples = []
        # with open(file_path, 'r', encoding="utf-8") as f:
        for line in file_lists:
            line=line.strip()
            data = json.loads(line)
            
            video_start=0.0
            video_end=None
            if 'start_time' in data:
                if data['start_time']:
                    video_start=data['start_time']
                if data['end_time']:
                    video_end=data['end_time']


            ques=data['conversations'][0]['value']
            ques=ques.replace('<video>\n','')

            example = {
                "problem": {"question":ques},
                "solution": {"answer":data['conversations'][1]['value']},
                "video": data['video'],
                "video_start":video_start,
                "video_end":video_end,
            }

            examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:5])
        dataset = Dataset.from_list(examples)

        dataset.client = None
        def __getitem__(self, idx): # Define getitem within the scope where dataset is available

            example = dataset[idx]
            data_to_return = {k: v for k, v in example.items()} # Create a copy to avoid modifying original dataset
            try:
                messages = [{"role": "user", "content": [{"type": "video", "video": example["video"][0],'video_start':example["video_start"][0],'video_end':example["video_end"][0],"total_pixels": 3584 * 28 * 28, "min_pixels": 16 * 28 * 28,},]}]
                
                image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, client=self.client)
                fps_inputs = video_kwargs['fps']
                # # data_to_return["image_inputs"] = [torch.load(os.path.join(example["video_path"][0], "image_inputs.pt"))]
                data_to_return["video_inputs"] = [video_inputs]
                # with open(os.path.join(example["video_path"][0], "video_kwargs.json"), 'r') as f:
                data_to_return["video_kwargs"] = [video_kwargs]
            except Exception as e:
                print(f"Warning: Error loading preprocessed data from {example['video'][0]}, falling back to video_path. Error: {e}")

                print(idx)
                if isinstance(idx, list):
                    idx=[ii+1 for ii in idx]
                else:
                    idx = idx + 1
                return self.__getitem__(idx)

            return data_to_return

        dataset.__getitem__ = __getitem__.__get__(dataset, Dataset) # Bind getitem to the dataset

        return dataset


    combine_data_lists=[]
    ds_collections = json.loads(open(train_data_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        data_path = ds_collections[ds_name]["annotation"]
        repeat_time = ds_collections[ds_name]['repeat_time']
        curr_data_list=read_jsonldataset(data_path,repeat_time)
        combine_data_lists.extend(curr_data_list)

    train_dataset = create_dataset_from_json(combine_data_lists, "train")
    return DatasetDict({"train": train_dataset})
    #eval_dataset = create_dataset_from_json(eval_data_path, "eval")
    #@return DatasetDict({"train": train_dataset, "eval": eval_dataset})

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset, now handles both raw and preprocessed data
    dataset = load_json_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.video_folder,
        # script_args.preprocessed_data_path # Pass preprocessed_data_path
    )




    if not training_args.use_vllm:
        trainer_cls = Qwen2VLGRPOTrainer
    else:
        raise NotImplementedError
    
    print("using: ", trainer_cls)

    # from peft import LoraConfig, get_peft_model

    # lora_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     inference_mode=False,
    #     r=64,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    # )

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    # trainer.train()
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)