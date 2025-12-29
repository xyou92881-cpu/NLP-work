项目介绍
本项目是一个基于 BART 预训练模型的口语化政务问题自动规范回答系统，核心功能是将用户以口语化方式提出的政务咨询（如 “社保断缴了能补交吗？”），自动转换为符合政务场景规范、严谨、正式的回答文本。
项目适用于政务服务平台的智能问答模块，可实现政务咨询的自动化、标准化回复，提升政务服务效率与规范化水平。
项目目录结构
plaintext
自然语言处理项目/
├─ assess.py                  # 模型评估脚本（批量测试+综合指标计算）
├─ datapreprocess.py          # 数据预处理脚本（清洗+编码+划分数据集）
├─ train.py                   # 模型训练脚本（BART模型微调+最优模型保存）
├─ train_data.csv             # 训练集（200条口语-正式语平行语料）
├─ val_data.csv               # 验证集（30条口语-正式语平行语料）
├─ test_data.csv              # 测试集（69条口语-正式语平行语料）
环境依赖
plaintext
python==3.9.13
torch==1.12.1
transformers==4.28.1
pandas==1.5.3
nltk==3.8.1
jieba==0.42.1
安装依赖：
bash
运行
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
快速使用
1. 训练模型
运行train.py脚本，自动完成数据加载、模型微调、最优模型保存：
bash
运行
python train.py
训练完成后，最优模型会保存至best_bart_style_transfer/目录。
2. 评估模型
运行assess.py脚本，自动加载训练好的模型，对test_data.csv进行批量评估，输出综合合格率、语义准确性等指标：
bash
运行
python assess.py
数据集说明
本项目使用的政务平行语料库包含 269 条有效样本，覆盖 6 类高频政务业务场景：
社保类（如 “社保断缴了能补交吗？”）
工商类（如 “营业执照丢了咋补办？”）
公积金类（如 “公积金能取出来租房吗？”）
医保类（如 “医保异地就医怎么报销？”）
不动产类（如 “不动产权证怎么办理？”）
婚姻登记类（如 “结婚证丢了能补办吗？”）
数据集按 75:10:15 划分为：
训练集（train_data.csv）：200 条
验证集（val_data.csv）：30 条
测试集（test_data.csv）：69 条
