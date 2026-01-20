import json
import re
from pprint import pprint
import pandas as pd

import torch
from datasets import Dataset
import evaluate
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers import Trainer
from torch.optim import AdamW

# =========================
# SETUP
# =========================
now = datetime.now()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# =========================
# Llama 3.1 8B Instruct
# =========================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_DIR = "./outputs"

train_csv_file = "./refactored_train.csv"
test_csv_file = "./refactored_test.csv"

# PROMPT
DEFAULT_SYSTEM_PROMPT = """You are an expert in software engineering and technical debt.
Your task is to classify a single source code comment into exactly ONE Self-Admitted Technical Debt (SATD) category.
Choose ONE label among: DEFECT, DESIGN, DOCUMENTATION,  IMPLEMENTATION, or TEST.
Return ONLY the label. Do NOT add explanations, punctuation, or extra text.""".strip()

AUTH_TOKEN = ""

# =========================
# DATASET
# =========================
train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)

train_df.drop(train_df[train_df["classification"] == "WITHOUT_CLASSIFICATION"].index, inplace=True)
test_df.drop(test_df[test_df["classification"] == "WITHOUT_CLASSIFICATION"].index, inplace=True)

print(f"The number of training data: {len(train_df.index)}")
print(f"The number of testing data: {len(test_df.index)}")

train_df = train_df.dropna(subset=["classification", "commenttext"])
test_df = test_df.dropna(subset=["classification", "commenttext"])

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# =========================
# PROMPT
# =========================
def generate_training_prompt(readme: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""### Instruction: {system_prompt}

### Input:
{readme.strip()}

### Response:
{summary}
""".strip()

def process_description(s: str) -> str:
    if s.endswith("."):
        s = s[:-1]
    s = re.sub(r"\. ", ", ", s)
    return s + "."

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"#+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)

def generate_sample_with_prompt(entry):
    readme = clean_text(entry["commenttext"])
    description = process_description(entry["classification"])
    return {
        "formatted_readme": readme,
        "summary": description,
        "prompt_text": generate_training_prompt(readme, description),
    }

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_sample_with_prompt)
        .remove_columns(["projectname", "classification", "commenttext"])
    )

example = generate_sample_with_prompt(train_dataset[0])

processed_train_dataset = process_dataset(train_dataset)

# =========================
# MODELLO + TOKENIZER (QLoRA)
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
        trust_remote_code=True,
        device_map="auto",
        use_auth_token=AUTH_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=AUTH_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False

# =========================
# LoRA
# =========================
lora_r = 16
lora_alpha = 64
lora_dropout = 0.1
lora_target_modules = [
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj",
]

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model = model.half()

# =========================
# TRAINING
# =========================
"""
training_arguments = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=3,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="none",
    save_safetensors=True,
    lr_scheduler_type="cosine",
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
"""

tokenizer.save_pretrained("./tokenizer")
tokenizer.model_max_length = 512

training_arguments = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=1e-4,
    fp16=False,
    max_grad_norm=0.3,
    num_train_epochs=3,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="none",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
    bf16=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=processed_train_dataset,
    formatting_func=lambda x: x["prompt_text"],
    args=training_arguments,
)

trainer.train()
trainer.save_model()

model = PeftModel.from_pretrained(model, OUTPUT_DIR)

# =========================
# TESTING PROMPT
# =========================
def generate_testing_prompt(readme: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""### Instruction: {system_prompt}

### Input:
{readme.strip()}

### Response:
""".strip()

examples = []
for entry in test_dataset:
    readme = clean_text(entry["commenttext"])
    description = entry["classification"]

    examples.append(
        {
            "formatted_readme": readme,
            "summary": description,
            "prompt_text": generate_testing_prompt(readme),
        }
    )

result_df = pd.DataFrame(examples)

# =========================
# GENERATION
# =========================
def summarize(model, text: str):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.0001)
    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

def correct_answer(response: str):
    return response.strip().split("\n")[0]

def generate_summary(prompt_text: str):
    raw_summary = summarize(model, prompt_text)
    corrected_summary = correct_answer(raw_summary)
    return corrected_summary

# =========================
# RUN INFERENCE
# =========================
result_list = []
for x in result_df["prompt_text"]:
    answer = ""
    try:
        answer = generate_summary(x)
    except Exception:
        print(x)
    result_list.append(answer)

print("RESULT COMPUTED")

result_df["generated_summary"] = result_list
result_df.to_csv(f"{OUTPUT_DIR}/compared_results_LLAMA.csv", index=False)

# =========================
# Precision/Recall/F1
# =========================

LABELS = ["DEFECT", "DESIGN", "DOCUMENTATION", "IMPLEMENTATION", "TEST"]

def normalize_multiclass(x: str) -> str:
    # Normalize the model’s output to ONE of the 5 classes.
    if x is None:
        return "IMPLEMENTATION"

    x = str(x).strip().upper()
    x = x.split("\n")[0].strip()
    x = re.sub(r"[^A-Z_ ]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    for lab in LABELS:
        if lab in x:
            return lab

    return "IMPLEMENTATION"


def normalize_gold_label(x: str) -> str:
# Normalize the gold label from the CSV
    if x is None:
        return "IMPLEMENTATION"
    x = str(x).strip().upper()
    x = x.replace(".", "").strip()
    return x

y_true = [normalize_gold_label(x) for x in result_df["summary"].tolist()]
y_pred = [normalize_multiclass(x) for x in result_df["generated_summary"].tolist()]

y_true = [x if x in LABELS else "IMPLEMENTATION" for x in y_true]

# =========================
# METRICS
# =========================
precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=LABELS,
    average=None,
    zero_division=0
)

p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

accuracy = accuracy_score(y_true, y_pred)

# =========================
# CSV
# =========================
metrics_rows = []

for i, lab in enumerate(LABELS):
    metrics_rows.append({
        "class": lab,
        "precision": precision[i],
        "recall": recall[i],
        "f1": f1[i],
        "support": int(support[i]),
    })

metrics_rows.append({
    "class": "macro_avg",
    "precision": p_macro,
    "recall": r_macro,
    "f1": f1_macro,
    "support": int(sum(support)),
})

metrics_rows.append({
    "class": "weighted_avg",
    "precision": p_weighted,
    "recall": r_weighted,
    "f1": f1_weighted,
    "support": int(sum(support)),
})

metrics_rows.append({
    "class": "accuracy",
    "precision": accuracy,
    "recall": accuracy,
    "f1": accuracy,
    "support": int(sum(support)),
})

metrics_df = pd.DataFrame(metrics_rows)

metrics_df.to_csv(
    f"{OUTPUT_DIR}/RQ3_metrics_precision_recall_f1_llama.csv",
    index=False
)

print(metrics_df)

later = datetime.now()
print("Total time (s):", (later - now).total_seconds())

