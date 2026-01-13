# test_generation.py
import torch
from llava.model.builder import load_pretrained_model
from PIL import Image

# Load model
model_path = "checkpoints/train_perspective_annealing-llava-v1.5-13b-task-lora/checkpoint-5560"

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path,
    "liuhaotian/llava-v1.5-13b",
    model_name,
    perspective_data=True
)

# Simple prompt
prompt = "Describe this image:"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

# Load a test image
image = Image.open("../data/evals/perspective_taking/images/left_cube_0.jpeg").convert('RGB')
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].cuda()

# Generate
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0),
        do_sample=True,
        temperature=0.7,
        max_new_tokens=100
    )

output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Generated text: {output}")

