import json
import re
from pprint import pprint
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

now = datetime.now()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# =========================
# Llama 3.1 8B Instruct
# =========================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_DIR = "./readme_summarization"

train_csv_file = "./refactored_train.csv"
test_csv_file = "./refactored_test.csv"

# PROMPT
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant specialized in classifying code comments.
Your task is to analyze the code comments.
Classify the following code comment as DEFECT, DESIGN, DOCUMENTATION, IMPLEMENTATION, or TEST.
Use just one class. Do not include any additional text.""".strip()

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

# =========================
# TRAINING
# =========================
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
# ROUGE
# =========================
metric = load_metric("rouge")
result = metric.compute(
    predictions=result_df["generated_summary"].to_list(),
    references=result_df["summary"].to_list(),
    use_stemmer=True,
)

result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
result = {k: round(v, 4) for k, v in result.items()}

json_object = json.dumps(result, indent=4)

with open(f"{OUTPUT_DIR}/rouge_results.json", "w") as outfile:
    outfile.write(json_object)


refactor_predictions = [s.split(".")[0] for s in result_df["generated_summary"].to_list()]
result1 = metric.compute(
    predictions=refactor_predictions,
    references=result_df["summary"].to_list(),
    use_stemmer=True,
)

result1 = {key: value.mid.fmeasure * 100 for key, value in result1.items()}
result1 = {k: round(v, 4) for k, v in result1.items()}

print(result1)

later = datetime.now()
print(str((later - now).total_seconds()))
