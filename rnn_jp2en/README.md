# 1 如何训练
## 1.1 词向量
运行：```python train_embd.py```

训练好的embedding会被保存在embd中，日语和英语的embedding分别为jp_embedding.pth和en_embedding.pth。

## 1.2 RNN模型
运行：```python train_rnn.py```

训练好的encoder和decoder的参数会被保存在model中，分别为encoder_params.pth和decoder_params.pth。

# 2 如何测试
运行：```python test_rnn.py```

test_rnn.py会加载训练好的参数encoder_params.pth和decoder_params.pth并输出当前模型在SPLIT所指定的数据集上的Loss、PPL、BLEU（比如SPLIT='train'时就会输出模型在训练集上的表现）。

# 3 如何生成翻译
运行：```python test_rnn_sample.py```

运行前在small_sample这个list中填入想要翻译的日文句子，运行后便会按序输出模型翻译后的英文句子。

注：环境配置在requirements.txt中。