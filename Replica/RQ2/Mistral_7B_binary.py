import json
import re
import pandas as pd

import torch
from datasets import Dataset, load_metric
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

# =========================
# SETUP GENERALE
# =========================
now = datetime.now()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# >>> MISTRAL 7B INSTRUCT <<<
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

OUTPUT_DIR = "./readme_summarization"
train_csv_file = "./refactored_train.csv"
test_csv_file = "./refactored_test.csv"

DEFAULT_SYSTEM_PROMPT = """
Classify the following text as YES or NO. Use just one class.
""".strip()

# =========================
# DATASET
# =========================
train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

train_df = train_df.dropna(subset=["classification", "commenttext"])
test_df = test_df.dropna(subset=["classification", "commenttext"])

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# =========================
# PROMPT
# =========================
def generate_training_prompt(readme, summary, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return f"""### Instruction: {system_prompt}

### Input:
{readme.strip()}

### Response:
{summary}
""".strip()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"#+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)

def generate_sample_with_prompt(entry):
    readme = clean_text(entry["commenttext"])
    label = "NO." if entry["classification"] == "WITHOUT_CLASSIFICATION" else "YES."
    return {
        "prompt_text": generate_training_prompt(readme, label),
        "summary": label,
    }

def process_dataset(data):
    return (
        data.shuffle(seed=42)
        .map(generate_sample_with_prompt)
        .remove_columns(["projectname", "classification", "commenttext"])
    )

processed_train_dataset = process_dataset(train_dataset)

# =========================
# QLoRA: MODELLO + TOKENIZER
# =========================
def create_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False

# =========================
# LoRA CONFIG
# =========================
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# =========================
# TRAINING
# =========================
training_arguments = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    output_dir=OUTPUT_DIR,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=processed_train_dataset,
    peft_config=peft_config,
    dataset_text_field="prompt_text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()
trainer.save_model()

# =========================
# LOAD MODEL FINETUNED
# =========================
model = PeftModel.from_pretrained(model, OUTPUT_DIR)

# =========================
# TESTING / INFERENCE
# =========================
def generate_testing_prompt(readme, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return f"""### Instruction: {system_prompt}

### Input:
{readme.strip()}

### Response:
""".strip()

examples = []
for entry in test_dataset:
    readme = clean_text(entry["commenttext"])
    label = "NO." if entry["classification"] == "WITHOUT_CLASSIFICATION" else "YES."
    examples.append(
        {
            "summary": label,
            "prompt_text": generate_testing_prompt(readme),
        }
    )

result_df = pd.DataFrame(examples)

def summarize(model, text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0001,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

def normalize_answer(ans):
    return ans.strip().split("\n")[0]

predictions = []
for p in result_df["prompt_text"]:
    try:
        pred = normalize_answer(summarize(model, p))
    except Exception:
        pred = ""
    predictions.append(pred)

result_df["generated_summary"] = predictions
result_df.to_csv(f"{OUTPUT_DIR}/compared_results_Mistral.csv", index=False)

# =========================
# ROUGE (come nel tuo script)
# =========================
metric = load_metric("rouge")
result = metric.compute(
    predictions=result_df["generated_summary"].tolist(),
    references=result_df["summary"].tolist(),
)

result = {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
print(result)

later = datetime.now()
print("Total time (s):", (later - now).total_seconds())
