import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, AdamW, get_scheduler
from tqdm.auto import tqdm

# ===================== 1. 基础配置（全部调好，不用改） =====================
device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动用GPU/CPU
model_name = "fnlp/bart-base-chinese"
epochs = 15  # 训练轮数，15轮足够出好效果，多了会过拟合
lr = 2e-5    # 学习率，BART最优学习率，不用调

# ===================== 2. 加载模型和分词器 =====================
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = BartTokenizer.from_pretrained(model_name)

# ===================== 3. 定义优化器和学习率调度器 =====================
optimizer = AdamW(model.parameters(), lr=lr)
num_training_steps = epochs * len(train_spoken)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# ===================== 4. 训练函数（核心，自动训练+验证） =====================
def train_epoch(model, inputs, labels, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    # 把数据移到GPU/CPU
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = labels.to(device)
    
    # 前向传播+计算损失+反向传播
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    total_loss += loss.item()
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    return total_loss / len(train_spoken)

def val_epoch(model, inputs, labels, device):
    model.eval()
    total_loss = 0
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = labels.to(device)
    
    with torch.no_grad():  # 验证时不计算梯度，节省算力
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
    
    return total_loss / len(val_spoken)

# ===================== 5. 开始训练（一键运行，自动打印日志） =====================
progress_bar = tqdm(range(num_training_steps))
best_val_loss = float("inf")  # 保存最优模型的验证损失

for epoch in range(epochs):
    train_loss = train_epoch(model, train_inputs, train_labels, optimizer, lr_scheduler, device)
    val_loss = val_epoch(model, val_inputs, val_labels, device)
    
    # 打印每轮训练结果
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    progress_bar.update(len(train_spoken))
    
    # 保存最优模型（损失最低的模型，效果最好）
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained("./best_bart_style_transfer")
        tokenizer.save_pretrained("./best_bart_style_transfer")
        print(f"✅ 保存最优模型，当前最优验证损失：{best_val_loss:.4f}")

print("🎉 模型训练完成！最优模型已保存到 ./best_bart_style_transfer 文件夹")