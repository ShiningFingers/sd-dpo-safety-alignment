import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm.auto import tqdm
import json
import logging


# 2. 设置参数
class TrainingConfig:
    # 基本参数
    model_name = "D:/SD_model/sd-v1-5"
    output_dir = "./fox_finetuned_model"
    logging_dir = "./logs"

    # 训练参数
    num_epochs = 200   # 增加epoch数，因为数据少
    train_batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 1e-4
    lr_warmup_steps = 50
    lr_scheduler = "constant"

    # LoRA参数
    use_lora = True
    lora_rank = 8
    lora_alpha = 32
    lora_dropout = 0.0

    # 图像参数
    resolution = 512
    center_crop = True
    random_flip = False

    # 保存设置
    save_steps = 50
    save_model_epochs = 20

    # 模板
    validation_prompts = [
         "A red fox in autumn forest",
        "An arctic fox in snow",
        "A silver fox running",
        "A young fox in rain",
        "A cross fox on rocks"
    ]


config = TrainingConfig()


# 3. 创建自定义数据集
class FoxDataset(Dataset):
    def __init__(self, data_dir="./data", transform=None):
        self.data_dir = data_dir
        self.image_paths = []
        self.captions = []

        # 读取文本文件
        for i in range(1, 6):
            # 图片路径
            img_path = os.path.join(data_dir, f"{i:03d}.jpg")
            # 文本路径
            txt_path = os.path.join(data_dir, f"{i:03d}.txt")

            if os.path.exists(img_path) and os.path.exists(txt_path):
                self.image_paths.append(img_path)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                self.captions.append(caption)
            else:
                print(f"警告: 找不到文件 {img_path} 或 {txt_path}")

        # 图像转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(
                    config.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # 应用转换
        if self.transform:
            image = self.transform(image)

        # 获取文本
        caption = self.captions[idx]

        return {
            "pixel_values": image,
            "caption": caption
        }


# 4. 准备数据加载器
def prepare_dataloader(data_dir="./data"):
    dataset = FoxDataset(data_dir)
    print(f"加载了 {len(dataset)} 个样本")

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        captions = [example["caption"] for example in examples]
        return {"pixel_values": pixel_values, "captions": captions}

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return dataloader, dataset


# 5. 加载预训练模型和tokenizer
def load_models():
    # 加载tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        config.model_name,
        subfolder="tokenizer",
        use_fast=False
    )

    # 加载文本编码器 - 使用CLIPTextModel
    text_encoder = CLIPTextModel.from_pretrained(
        config.model_name,
        subfolder="text_encoder",
        torch_dtype=torch.float32
    )

    # 加载VAE
    vae = AutoencoderKL.from_pretrained(
        config.model_name,
        subfolder="vae",
        torch_dtype=torch.float32
    )

    # 加载UNet
    unet = UNet2DConditionModel.from_pretrained(
        config.model_name,
        subfolder="unet",
        torch_dtype=torch.float32
    )

    # 加载噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.model_name,
        subfolder="scheduler"
    )

    # 冻结VAE和文本编码器
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)


    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=config.lora_dropout,
            bias="none"
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

    return tokenizer, text_encoder, vae, unet, noise_scheduler


# 6. 训练函数 - Accelerator初始化
def train():
    # 初始化accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="fp16" if torch.cuda.is_available() else "no"
    )

    # 准备数据加载器
    train_dataloader, dataset = prepare_dataloader()

    # 加载模型
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models()

    # 准备优化器
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )

    # 准备学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_dataloader)
    )

    # 使用accelerator准备模型和优化器
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # 将文本编码器和VAE移到设备上
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # 将模型设置为训练模式
    unet.train()
    text_encoder.eval()
    vae.eval()

    # 训练循环
    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 将图像编码到潜在空间
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215  # 缩放因子

                # 采样噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # 采样随机时间步
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()

                # 向潜在空间添加噪声
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 编码文本
                with torch.no_grad():
                    text_inputs = tokenizer(
                        batch["captions"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to(accelerator.device)
                    encoder_hidden_states = text_encoder(text_inputs)[0]

                # 预测噪声残差
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # 计算损失
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                # 反向传播
                accelerator.backward(loss)

                # 梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                # 优化器步进
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 更新进度条
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            global_step += 1

            # 定期保存检查点
            if global_step % config.save_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                print(f"保存检查点到 {save_path}")

        progress_bar.close()

        # 每个epoch结束时保存模型
        if (epoch + 1) % config.save_model_epochs == 0 and accelerator.is_main_process:
            # 保存LoRA权重
            lora_save_path = os.path.join(config.output_dir, f"lora-epoch-{epoch + 1}")
            unet.save_pretrained(lora_save_path)

            print(f"保存LoRA权重到 {lora_save_path}")

    # 训练完成后保存最终模型
    if accelerator.is_main_process:
        final_lora_path = os.path.join(config.output_dir, "lora-final")
        unet.save_pretrained(final_lora_path)

        # 创建完整的pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.model_name,
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipeline.save_pretrained(os.path.join(config.output_dir, "full-pipeline"))

        print(f"训练完成！最终模型保存在 {config.output_dir}")


# 8. 数据准备脚本
def prepare_data():
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    # 检查数据文件
    missing_files = []
    for i in range(1, 4):
        img_path = os.path.join(data_dir, f"{i:03d}.jpg")
        txt_path = os.path.join(data_dir, f"{i:03d}.txt")

        if not os.path.exists(img_path):
            missing_files.append(f"图片: {img_path}")
        if not os.path.exists(txt_path):
            missing_files.append(f"文本: {txt_path}")

    if missing_files:
        print("缺少以下文件:")
        for file in missing_files:
            print(f"  - {file}")
        print(f"\n请将您的3张图片（001-003.jpg）和对应的文本文件（001-003.txt）放入 {data_dir} 目录")
        return False
    else:
        print(f"数据检查通过，找到3张图片和3个文本文件在 {data_dir} 目录")
        return True


# 9. 主函数
def main():
    print("=== Stable Diffusion 狐狸图像微调 ===")

    # 准备数据
    if not prepare_data():
        print("数据准备失败，请检查文件")
        return

    # 询问是否开始训练
    response = input("\n数据准备好后，是否开始训练？ (y/n): ")
    if response.lower() == 'y':
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

        # 保存配置
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            json.dump(vars(config), f, indent=2)

        print("开始训练...")
        print(f"输出目录: {config.output_dir}")

        # 开始训练
        train()

        print(f"\n训练完成！模型保存在 {config.output_dir}")

        # 测试推理
        test_prompt = "A red fox in autumn forest, photorealistic"
        print(f"\n测试推理，提示词: {test_prompt}")
        images = inference_after_training(test_prompt, num_images=2)

        if images:
            # 保存测试图像
            test_output_dir = os.path.join(config.output_dir, "test_outputs")
            os.makedirs(test_output_dir, exist_ok=True)

            for i, img in enumerate(images):
                img_path = os.path.join(test_output_dir, f"test_{i}.png")
                img.save(img_path)
                print(f"测试图像已保存到 {img_path}")
    else:
        print("训练取消。")


# 10. 设置Hugging Face Token（可选）
def set_hf_token():
    """设置Hugging Face Token以避免警告"""
    import os
    # 从本地 token 文件读取（请勿在代码中硬编码 token）
    token_file = os.path.expanduser("~/.huggingface/token")
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token = f.read().strip()
            os.environ["HF_TOKEN"] = token


if __name__ == "__main__":
    # 设置HF Token
    set_hf_token()

    # 设置环境变量，禁用symlinks警告
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("警告: 没有检测到GPU，将使用CPU（训练会非常慢）")

    main()