import os
import argparse
import torch
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

def get_args():
    parser = argparse.ArgumentParser(description="Pretrain RoBERTa-medium on Wiki-103")
    parser.add_argument("--output_dir", type=str, default="checkpoints/roberta-medium-wiki103", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=20000, help="Warmup steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Per device batch size") # Increased for A100
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available, training will be very slow on CPU!")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Config for RoBERTa-medium
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=514,
        # position_embedding_type="relative_key_query",
        type_vocab_size=1,
    )
    
    model = RobertaForMaskedLM(config)
    
    # Load Data
    print("Loading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    # Preprocessing - use chunking approach for MLM pretraining
    # This is much more efficient than padding every example to max_length
    def tokenize_function(examples):
        # Tokenize without padding - we'll handle that in the collator
        return tokenizer(examples["text"], return_special_tokens_mask=True)
    
    def group_texts(examples):
        # Concatenate all texts and chunk into fixed-length blocks
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop the small remainder
        total_length = (total_length // args.max_seq_length) * args.max_seq_length
        # Split by chunks of max_len
        result = {
            k: [t[i : i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    print("Tokenizing dataset...")
    # Filter empty lines first to save time
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    
    print("Grouping texts into chunks...")
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
    )
    
    # No padding needed since all sequences are already max_seq_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, pad_to_multiple_of=None
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        save_steps=50000,
        save_total_limit=2,
        seed=args.seed,
        no_cuda=False,  # Explicitly enable CUDA
        fp16=torch.cuda.is_available(), # Use fp16 if cuda is available
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        report_to="none", # Disable wandb/mlflow for this script
        # Optionally enable gradient checkpointing to save memory and allow larger batches
        # gradient_checkpointing=True,
    )
    
    print(f"Training will use device: {training_args.device}")
    print(f"FP16 training: {training_args.fp16}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    
    # Check if a checkpoint exists
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(args.output_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")
    
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
