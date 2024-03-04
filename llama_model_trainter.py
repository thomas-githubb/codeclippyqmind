from datasets import load_dataset
import torch  # Import torch first
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from huggingface_hub import login 
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import os

# NOTE: SETTING UP MODEL AND TRAINING
model_name = "NousResearch/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "llama-2-7b-miniguanaco"

# NOTE: edit dataset code
loaded_dataset_1 = load_dataset(path="semeru/code-code-CodeCompletion-TokenLevel-Python", split="train")

loaded_dataset_2 = load_dataset("code_x_glue_cc_code_completion_line", 'python')

def new_prompt_1(prompt):
    new_prompt = f"<s>[INST]<<SYS>>{{ system_prompt }}<</SYS> {prompt['text']}[/INST]"
    prompt['text'] = new_prompt
    return prompt

def new_prompt_2(prompt):
    new_prompt = f"<s>[INST]<<SYS>>{{ system_prompt }}<</SYS> {prompt['input']}[/INST]"
    del prompt['input']
    prompt['text'] = new_prompt
    return prompt

dataset_python_1 = loaded_dataset_1.map(new_prompt_1)
dataset_python_2 = loaded_dataset_2.map(new_prompt_2)

dataset = concatenate_datasets([dataset_python_1])

fraction_of_data = 0.001 
dataset = dataset.select(range(int(len(dataset) * fraction_of_data)))

# Define QLoRA and bitsandbytes parameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = False
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Define TrainingArguments parameters
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25

# Define SFT parameters
max_seq_length = None
packing = False
device_map = {"": 0}


# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(load_in_4bit=use_4bit, bnb_4bit_quant_type=bnb_4bit_quant_type, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=use_nested_quant)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Adjust per_device_train_batch_size and gradient_accumulation_steps
per_device_train_batch_size = 1  
gradient_accumulation_steps = 4  

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps, 
    save_steps=save_steps, logging_steps=logging_steps,
    learning_rate=learning_rate, weight_decay=weight_decay,  max_grad_norm=max_grad_norm,
    max_steps=max_steps, warmup_ratio=warmup_ratio, group_by_length=group_by_length, lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    fp16=fp16, bf16=bf16, 
    
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto')
model = model.to('mps')
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(lora_alpha=lora_alpha, lora_dropout=lora_dropout, r=lora_r, bias="none", task_type="CAUSAL_LM")

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model, train_dataset=dataset, peft_config=peft_config, dataset_text_field="text", max_seq_length=max_seq_length,
    tokenizer=tokenizer, args=training_arguments, packing=packing, 
)

torch.mps.empty_cache()

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)



token = os.environ.get('token')

login(token=token)

trainer.model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)


from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login 

config = PeftConfig.from_pretrained("mthw/llama-2-7b-miniguanaco")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
model = PeftModel.from_pretrained(model, "mthw/llama-2-7b-miniguanaco")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", trust_remote_code=True)
model = model.merge_and_unload()


login(token=token)
model.push_to_hub("llama-2-7b-miniguanaco", use_temp_dir=False)
tokenizer.push_to_hub("llama-2-7b-miniguanaco", use_temp_dir=False)

