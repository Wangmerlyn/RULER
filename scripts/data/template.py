# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

Templates = {
    'base': "{task_template}",

    'meta-chat': "[INST] {task_template} [/INST]",

    'vicuna-chat': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {task_template} ASSISTANT:",

    'lwm-chat': "You are a helpful assistant. USER: {task_template} ASSISTANT: ",

    'command-r-chat': "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{task_template}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",

    'chatglm-chat': "[gMASK]sop<|user|> \n {task_template}<|assistant|> \n ",
    
    'RWKV': "User: hi\n\nAssistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it\n\nUser: {task_template}\n\nAssistant:",

    'Phi3': "<|user|>\n{task_template}<|end|>\n<|assistant|>\n",
    'Phi3.1': "<|user|> {task_template} <|end|> <|assistant|>",
    'meta-llama3': "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{task_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "qwen-math": "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{{{}}}}.<|im_end|>\n<|im_start|>user\n{task_template}<|im_end|>\n<|im_start|>assistant\n",
    'llama_distill_r1': "<｜begin▁of▁sentence｜><｜User｜>{task_template}<｜Assistant｜><think>\\n",
    'qwen': "'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{task_template}<|im_end|>\n<|im_start|>assistant\n'",
    "orz": 'A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{{{}}}} tag.\nThis is the problem:\n{task_template}\nAssistant: ',
    'qwq':'<|im_start|>user\n{task_template}<|im_end|>\n<|im_start|>assistant\n<think>\n',
    "qwen_orz_boxed": "<|im_start|>system\nA conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within \\boxed{{{{}}}} tags, respectively, i.e., <think> reasoning process here </think> \\boxed{{{{answer here}}}}.<|im_end|>\n<|im_start|>user\n{task_template}<|im_end|>\n<|im_start|>assistant\n<think>",
    "think_model_api":'{task_template}\nFormat your response as follows: "Therefore, the answer is (insert answer here)".'
}