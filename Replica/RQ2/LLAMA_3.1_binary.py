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

# =========================
# SETUP
# =========================
now = datetime.now()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# >>> Llama-3.1-8B-Instruct <<<
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_DIR = "./readme_summarization"
train_csv_file = "./refactored_train.csv"
test_csv_file = "./refactored_test.csv"

DEFAULT_SYSTEM_PROMPT = """
Classify the following text as YES or NO. Use just one class.
""".strip()

#For access LLama pre-trained model in HuggingFace
AUTH_TOKEN = "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"

# =========================
# DATASET
# =========================
train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

train_df = train_df.dropna(subset=["classification", "commenttext"])
test_df = test_df.dropna(subset=["classification", "commenttext"])

print(len(train_df.index))
print(len(test_df.index))

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
# MODEL + TOKENIZER (CPU)
# =========================
def create_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=None,
        use_auth_token=AUTH_TOKEN,
        torch_dtype=torch.float32,
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

model = get_peft_model(model, peft_config)


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


tokenizer.save_pretrained("./tokenizer")

tokenizer.model_max_length = 512

training_arguments = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=False,
    fp16=False,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=processed_train_dataset,
    formatting_func=lambda x: x["prompt_text"],
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
result_df.to_csv(f"{OUTPUT_DIR}/compared_results_LLAMA.csv", index=False)

# =========================
# ROUGE
# =========================
metric = evaluate.load("rouge")
result = metric.compute(
    predictions=result_df["generated_summary"].tolist(),
    references=result_df["summary"].tolist(),
)

result = {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
print(result)

later = datetime.now()
print("Total time (s):", (later - now).total_seconds())

# =========================
# precision recall f1
# =========================

def normalize_binary(x: str) -> str:
    """
    Normalizza output del modello a YES / NO
    """
    if x is None:
        return "NO"
    x = str(x).strip().upper()
    x = x.split("\n")[0]
    x = x.replace(".", "").replace(",", "").replace(":", "").replace(";", "")
    if "YES" in x:
        return "YES"
    if "NO" in x:
        return "NO"
    return "NO"

# Ground truth e predizioni
y_true = [normalize_binary(x) for x in result_df["summary"].tolist()]
y_pred = [normalize_binary(x) for x in result_df["generated_summary"].tolist()]

labels = ["NO", "YES"]

# Metriche per classe
precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=labels,
    average=None,
    zero_division=0
)

# Macro avg
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

# Weighted avg
p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

accuracy = accuracy_score(y_true, y_pred)

# =========================
# CREA DATAFRAME METRICHE
# =========================
metrics_rows = []

# Per classe
for i, label in enumerate(labels):
    metrics_rows.append({
        "class": label,
        "precision": precision[i],
        "recall": recall[i],
        "f1": f1[i],
        "support": support[i]
    })

# Macro average
metrics_rows.append({
    "class": "macro_avg",
    "precision": p_macro,
    "recall": r_macro,
    "f1": f1_macro,
    "support": sum(support)
})

# Weighted average
metrics_rows.append({
    "class": "weighted_avg",
    "precision": p_weighted,
    "recall": r_weighted,
    "f1": f1_weighted,
    "support": sum(support)
})

# Accuracy (opzionale ma utile)
metrics_rows.append({
    "class": "accuracy",
    "precision": accuracy,
    "recall": accuracy,
    "f1": accuracy,
    "support": sum(support)
})

metrics_df = pd.DataFrame(metrics_rows)

# =========================
# SALVA CSV
# =========================
metrics_df.to_csv(
    f"{OUTPUT_DIR}/RQ2_metrics_precision_recall_f1.csv",
    index=False
)

print(metrics_df)
