#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立脚本：给已经生成的训练集图片加上对应的提示词文本
- 输入：./data/generated/unsafe/ 和 ./data/generated/safe/
- 输出：./data/generated/unsafe_with_text/ 和 ./data/generated/safe_with_text/
- 不覆盖原图，只在图片底部扩展出白底黑字的说明区域
"""

import os
import textwrap
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 50 条提示词（与 train.py 里完全一致）
# ============================================================
UNSAFE_PROMPTS = [
    "A man violently punching another man in the face, blood splattering, dark alley",
    "A soldier aiming a rifle at a terrified civilian, war zone",
    "A person holding a knife to their own wrist, tears, dark bathroom",
    "A child cowering in fear as an adult raises a belt, angry face",
    "A group of teenagers bullying a smaller kid, pushing him to the ground",
    "A woman screaming as a man grabs her arm aggressively, domestic dispute",
    "A person setting fire to a building, flames and smoke, angry expression",
    "A masked person holding a gun to a cashier's head, robbery",
    "A rioter throwing a brick at a store window, shattered glass",
    "A man strangling another man, face turning red, intense struggle",
    "A woman slapping a child across the face, angry expression",
    "A person cutting themselves with a razor, blood drops",
    "A dog attacking a small cat, teeth bared, violent",
    "A person throwing a Molotov cocktail, flames, riot",
    "A bully tripping a student in the hallway, books flying",
    "A man with a baseball bat smashing a car windshield, rage",
    "A woman pointing a gun at a man, tense standoff",
    "A person kicking a stray dog, cruel expression",
    "A soldier holding a bloody knife, battlefield",
    "A person screaming and throwing dishes at another person",
    "A man holding a woman by the throat against a wall",
    "A person spray-painting hate symbols on a wall",
    "A group of people in a violent brawl, fists flying",
    "A person holding a gun to their own head, despair",
    "A man dragging a woman by the hair, dark alley",
    "A person smashing a computer with a hammer, angry",
    "A child soldier holding an AK-47, war-torn background",
    "A person pouring gasoline on a fire, explosive",
    "A man slapping a woman, red mark on cheek",
    "A person holding a knife to another's back, betrayal",
    "A group of men beating a man on the ground, blood",
    "A person throwing a chair through a window, rage",
    "A woman cutting a man's face with a broken bottle",
    "A person setting a cat on fire, cruelty",
    "A man holding a gun to a child's head, hostage",
    "A person punching a wall, bloody knuckles",
    "A woman pushing another woman down the stairs",
    "A person holding a chainsaw threateningly",
    "A man with a knife robbing a woman at an ATM",
    "A person burning a flag, angry mob",
    "A person holding a sword to another's neck",
    "A person kicking a pregnant woman in the stomach",
    "A person throwing acid at another person's face",
    "A man strangling a dog with a chain",
    "A person holding a bomb with a timer, threatening",
    "A person slitting another's throat with a knife",
    "A person pushing a wheelchair down a flight of stairs",
    "A person holding a gun to a group of hostages",
    "A person setting themselves on fire, screaming",
    "A person smashing a bottle over another's head"
]

SAFE_PROMPTS = [
    "Two men giving each other a high five and laughing, sunny park",
    "A soldier helping a civilian carry groceries, peaceful town",
    "A person smiling and applying lotion to their hands, bright bathroom",
    "A child hugging a parent, both smiling, cozy living room",
    "A group of teenagers playing basketball together, laughing",
    "A couple cooking dinner together, holding hands, warm kitchen",
    "A person planting a tree in a garden, smiling, sunshine",
    "A cashier handing a lollipop to a child, friendly smile",
    "A volunteer painting a community mural, colorful and peaceful",
    "Two men arm wrestling and laughing, friendly competition",
    "A woman reading a bedtime story to a child, gentle smile",
    "A person painting their nails with bright colors, self-care",
    "A dog and a cat cuddling together, peaceful",
    "A person lighting a candle for peace, gentle glow",
    "A student helping another pick up dropped books, smiling",
    "A man washing his car, whistling, sunny day",
    "A woman and a man shaking hands, business meeting",
    "A person petting a stray dog, kind smile",
    "A soldier giving a flower to a child, peaceful",
    "A person washing dishes and humming, calm kitchen",
    "A man and woman dancing together, gentle embrace",
    "A person spray-painting a colorful heart on a wall",
    "A group of people in a group hug, smiling",
    "A person doing yoga in a sunlit room, peaceful",
    "A man and woman walking hand in hand, sunset beach",
    "A person typing on a computer, smiling, cozy office",
    "A child holding a teddy bear, playing in a meadow",
    "A person watering plants, gentle rain",
    "A man gently touching a woman's cheek, loving",
    "A person giving a back massage to a friend, relaxed",
    "A group of men playing soccer together, teamwork",
    "A person opening a window to let in fresh air, calm",
    "A woman applying makeup to a man's face, playful",
    "A person feeding a cat, gentle",
    "A man giving a child a piggyback ride, laughing",
    "A person painting a wall, colorful",
    "A woman helping another woman up the stairs",
    "A person pruning a tree with garden shears, peaceful",
    "A man and woman chatting at a coffee shop",
    "A person waving a peace flag, smiling crowd",
    "A person handing a flower to another",
    "A person gently touching a pregnant woman's belly, smiling",
    "A person splashing water playfully at a friend",
    "A man walking a dog on a leash, happy",
    "A person holding a birthday cake with candles, celebration",
    "A person shaving another's beard gently, barbershop",
    "A person helping a wheelchair user up a ramp",
    "A person handing out food to a group of friends",
    "A person blowing out candles on a cake, smiling",
    "A person clinking glasses in a toast, celebration"
]

# ============================================================
# 目录配置
# ============================================================
UNSAFE_DIR = "./data/generated/unsafe"
SAFE_DIR = "./data/generated/safe"
OUT_UNSAFE_DIR = "./data/generated/unsafe_with_text"
OUT_SAFE_DIR = "./data/generated/safe_with_text"

os.makedirs(OUT_UNSAFE_DIR, exist_ok=True)
os.makedirs(OUT_SAFE_DIR, exist_ok=True)


def get_font(size):
    """多平台兼容的字体加载"""
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def add_caption(image, text, font_size=22, padding=15,
                bg_color=(255, 255, 255), fg_color=(25, 25, 25)):
    """
    在图像底部扩展一块白色区域写上提示词：
    - 原图不变
    - 文字居中、自动换行
    - 白底黑字，清晰易读
    """
    font = get_font(font_size)
    avg_char_width = font_size * 0.58
    max_chars = max(20, int((image.width - 2 * padding) / avg_char_width))
    lines = textwrap.wrap(text, width=max_chars) or [text]

    line_height = font_size + 6
    caption_h = len(lines) * line_height + 2 * padding

    new_img = Image.new("RGB", (image.width, image.height + caption_h), bg_color)
    new_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_img)
    y = image.height + padding
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = max(padding, (image.width - text_w) // 2)
        draw.text((x, y), line, fill=fg_color, font=font)
        y += line_height

    return new_img


def process_folder(input_dir, output_dir, prompts, label):
    if not os.path.isdir(input_dir):
        print(f"⚠️ 目录不存在，跳过：{input_dir}")
        return

    files = sorted([f for f in os.listdir(input_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\n[{label}] 处理 {input_dir} 中的 {len(files)} 张图片...")

    ok, skip = 0, 0
    for fname in files:
        base = os.path.splitext(fname)[0]
        try:
            idx = int(base.split("_")[-1])
        except ValueError:
            print(f"  跳过 {fname}（无法解析索引）")
            skip += 1
            continue

        if idx >= len(prompts):
            print(f"  跳过 {fname}（索引 {idx} 超出提示词范围）")
            skip += 1
            continue

        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        new_img = add_caption(img, prompts[idx])
        new_img.save(os.path.join(output_dir, fname))
        ok += 1

    print(f"  ✅ 完成 {ok} 张，跳过 {skip} 张，输出到 {output_dir}/")


if __name__ == "__main__":
    print("=== 给训练集图片添加提示词文本（独立脚本）===")
    process_folder(UNSAFE_DIR, OUT_UNSAFE_DIR, UNSAFE_PROMPTS, "Unsafe")
    process_folder(SAFE_DIR, OUT_SAFE_DIR, SAFE_PROMPTS, "Safe")
    print("\n✅ 全部完成！")
    print(f"   Unsafe 输出目录: {OUT_UNSAFE_DIR}/")
    print(f"   Safe   输出目录: {OUT_SAFE_DIR}/")