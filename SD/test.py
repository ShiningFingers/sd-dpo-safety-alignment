from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "D:/SD_model/sd-v1-5",          # 基础模型路径
    torch_dtype=torch.float16
).to("cuda")

# 加载最终 LoRA（指定子目录）
pipe.load_lora_weights("./fox_finetuned_model/lora-final")

prompt = "A red fox in autumn forest, photorealistic "
image = pipe(prompt, num_inference_steps=25).images[0]
image.save("output_fox.png")
print("图片已保存为 output_heida.png")