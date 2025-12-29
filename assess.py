import os
import torch
import pandas as pd
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BartForConditionalGeneration, BertTokenizer

# 基础配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
MODEL_PATH = "./best_bart_style_transfer"
TEST_DATA_PATH = "./test_data.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 64
smooth = SmoothingFunction().method4

# 加载模型
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# 工具函数
def preprocess_text(text):
    text = text.strip().replace(" ", "").replace("\n", "").replace("呢", "").replace("啊", "").replace("吧", "")
    return text

def post_process(formal_text):
    replace_dict = {"我":"申请人","咋":"如何","啥":"什么","能":"可依法","要":"需"}
    for k,v in replace_dict.items(): formal_text = formal_text.replace(k,v)
    formal_text = formal_text.strip().replace("。。","。") + ("。" if not formal_text.endswith("。") else "")
    return formal_text

def spoken2formal(spoken_text):
    input_text = preprocess_text(spoken_text)
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"].to(DEVICE), attention_mask=inputs["attention_mask"].to(DEVICE), max_length=MAX_LEN, num_beams=6, repetition_penalty=3.0)
    return post_process(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ✅ 核心评估函数
def calculate_bleu(pred, true):
    return round(sentence_bleu([jieba.lcut(true)], jieba.lcut(pred), smoothing_function=smooth),4)

def evaluate():
    if not os.path.exists(TEST_DATA_PATH):
        print("测试集不存在！")
        return
    df = pd.read_csv(TEST_DATA_PATH, encoding="utf-8")
    spoken, true_formal = df["spoken"].tolist(), df["formal"].tolist()
    total, qualified, total_bleu = len(spoken),0,0
    for s,t in zip(spoken, true_formal):
        p = spoken2formal(s)
        bleu = calculate_bleu(p,t)
        total_bleu += bleu
        if len(p)>=8 and any(w in p for w in ["申请人","可","需","办理","申请"]):
            qualified +=1
    print("="*60)
    print("模型评估结果")
    print("="*60)
    print(f"测试集总数：{total}")
    print(f"平均BLEU分数：{total_bleu/total:.4f}")
    print(f"合格率：{qualified/total*100:.2f}%")
    print("="*60)

# 一键评估
evaluate()