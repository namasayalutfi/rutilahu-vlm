import json
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer

DATASET_FILE = "../data/sft_dataset/train_vlm.jsonl"
IMAGE_DIR = "../data/raw_img"

# load dataset
data = []
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

dataset = Dataset.from_list(data)

# load model
model_id = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id)

def preprocess(example):
    image_path = f"{IMAGE_DIR}/{example['image']}"
    image = Image.open(image_path).convert("RGB")

    prompt = example["instruction"]
    answer = example["response"]

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs["labels"] = processor.tokenizer(answer).input_ids

    return inputs

dataset = dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir="vlm_model",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()