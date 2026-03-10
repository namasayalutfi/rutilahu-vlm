import json

input_file = "data/sft_dataset/train_vlm.jsonl"
output_file = "data/sft_dataset/train_qwen_vl.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(output_file, "w", encoding="utf-8") as out:
    for line in lines:
        item = json.loads(line)

        new_item = {
            "messages":[
                {
                    "role":"user",
                    "content":[
                        {"type":"image","image":f"raw_img/{item['image']}"},
                        {"type":"text","text":item["instruction"]}
                    ]
                },
                {
                    "role":"assistant",
                    "content":[
                        {"type":"text","text":item["response"]}
                    ]
                }
            ]
        }

        out.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print("Dataset converted to Qwen-VL format!")