import os
import argparse

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

try:
    from transformers import BitsAndBytesConfig
    USE_4BIT = torch.cuda.is_available()
except ImportError:
    USE_4BIT = False


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def format_instruction(sample):
    return f"""<s>[INST] Write a news article about the following topic: [/INST]
{sample["text"]}</s>"""

def load_base_model(base_id: str, kbit: bool = True):
    if kbit and torch.cuda.is_available():
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    return model

def finetune_model(args):
    # Load dataset from Hugging Face Hub
    if "/" in args.dataset:
        # If dataset path contains a slash, treat it as a Hugging Face dataset
        ds = load_dataset(args.dataset, split="train")
    else:
        # Otherwise, treat it as a local file
        ds = load_dataset("json", data_files=args.dataset)["train"]

    # base model to finetune
    model_id = args.base_model

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # load model and tokenizer
    model = load_base_model(model_id, kbit=args.use_qlora)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
        token=args.auth_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    # prepare model for training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, peft_config)
    
    # print the number of trainable model params
    print_trainable_parameters(model)

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # for causal-LM fine-tuning
    )

    training_args = TrainingArguments(
        output_dir=args.model_name,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=25,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.model_name if args.push_to_hub else None
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
        formatting_func=format_instruction,
    )

    # train
    trainer.train() 

    # save model
    trainer.save_model()

    if args.push_to_hub:
        trainer.model.push_to_hub(args.model_name)
        
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="neuralwork/fashion-style-instruct", 
        help="Path to local or HF dataset."
    )
    parser.add_argument(
        "--base_model", type=str, default="mistralai/Mistral-7B-v0.1", 
        help="HF hub id of the base model to finetune."
    )
    parser.add_argument(
        "--model_name", type=str, default="mistral-7b-style-instruct", help="Name of finetuned model."
    )
    parser.add_argument(
        "--auth_token", type=str, default=None, 
        help="HF authentication token, only used if downloading a private dataset."
    )
    parser.add_argument(
        "--push_to_hub", default=False, action="store_true", 
        help="Whether to push finetuned model to HF hub."
    )
    parser.add_argument(
        "--use_qlora", default=False, action="store_true", 
        help="Whether to use QLoRA for model quantization."
    )
    args = parser.parse_args()
    finetune_model(args)
