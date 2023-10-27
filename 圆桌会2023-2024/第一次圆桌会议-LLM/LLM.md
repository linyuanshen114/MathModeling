# LLM

# 1、原理介绍

### Transformer

Transformer 模型最主要的两个部分，编码器和解码器

![transformer](https://cdn.jsdelivr.net/gh/linyuanshen114/MathModeling@1.0.0/2023-2024 圆桌会/第一次圆桌会议-LLM/transformer.png)

编码器便是把你输入的文字进行量化，也就是转化为模型可以理解的数字语言。

embedding 简单说就是查表，在已经训练好的数据集中，每个词都对应一个向量，查出这个词所对应的向量

![token](token.png)

当然，一个句子里每个词位置不同，作用和代表的意思也不同，学者们提出了这样一个三角函数，它能很好的描述每个位置的一个信息。把位置向量再叠加进embeding里

![position encoding](position.png)

至此，我们得到了一个句子所对应的矩阵，也就对句子信息转化为了一个模型输入。

接着进入到了多头注意力层，简单理解便是，一个句子中有指代、修饰等等词与词之间的关系。单独地将每个token列为一个向量缺少了对句子的真正理解。这个层要做的就是将矩阵乘上一些特定的矩阵，生成三种特征矩阵，通过每个词生成的三种特征向量，对每个词之间的关系进行打分。
![](qkv.png)
![](thinking.jpg)
![](it.jpg)

经过多头注意力层后，我们就得到了多个这样的注意力矩阵。但是进入解码器之前要经过前馈神经网络处理，所以我们把它们拼起来，再用一个特殊的权重矩阵相乘。
就得到了输入进去的矩阵。

解码器便是类似的操作，将神经网络输入的向量在投影回每个token，看每个词在每个位置出现的概率，寻找到最优组合，转化为词句。就是ai给你的回答

# 2、国内外现用大模型概览

##  (1) ChatGPT

[chat.openai.com](chat.openai.com)

![](chatgpt_1.png)

![](chatpgt_2.png)

令人恐惧的语音生成和对话
[进来听气泡音ChatGPT骂人](https://www.bilibili.com/video/BV1dF41117oD/?spm_id_from=333.788.recommend_more_video.1&vd_source=a4ae0a3422ed0fc5f6330eca2de41249)

但是喜欢瞎说，并且容易自我怀疑

![](chatgpt_3.png)

![](chatgpt_4.png)

##  (2) New Bing

bing有点麻烦，你在中国会一直重定向到cn的bing网站，需要在设置里面调成外国的才有办法用。[水源](https://shuiyuan.sjtu.edu.cn/t/topic/141984)上有完整的注册教程


![bing](bing_1.png)

new bing有三种格式，并且会对内容进行网络上的查证和引用。相比于chatgpt来说更灵活和准确。创造力模块还有点小脾气，如果问太多问题或者一些情绪上的对话，他会回击你或者直接不干了。

![](bing_2.png)

##  (3) Bard

[https://bard.google.com/chat?hl=en](https://bard.google.com/chat?hl=en)

刚出来的时候比较拉胯，没太多使用

### (4) Copilot Chat

专注于代码的插件（实际演示）

[https://github.com/github-copilot/free_signup](https://github.com/github-copilot/free_signup)

要免费使用，需要通过学生认证[知乎教程](https://zhuanlan.zhihu.com/p/617001126)



##  (1) 文心一言

[https://yiyan.baidu.com/](https://yiyan.baidu.com/)

##  (2) 讯飞星火

[https://xinghuo.xfyun.cn/](https://xinghuo.xfyun.cn/)

##  (3) 通义千问

[http://qianwen.aliyun.com/](http://qianwen.aliyun.com/)

##  (4) 智谱清言

[https://chatglm.cn/](https://chatglm.cn/)

##  (5) MOSS

[https://moss.fastnlp.top/](https://moss.fastnlp.top/)

##  (6) Chatsjtu

[https://gpt.sjtu.edu.cn/](https://gpt.sjtu.edu.cn/)

## (X) Chirper

[https://chirper.ai/zh](https://chirper.ai/zh)

# 3、让ai表现更好

[https://platform.openai.com/docs/guides/gpt-best-practices](https://platform.openai.com/docs/guides/gpt-best-practices)

### 把问题说明清楚

#### 把细节说清楚

BAD：帮我写份简历

GOOD：我是一个即将从数学专业毕业的本科生，我想要申请到加里顿大学教育学硕士，我目前GPA 2.0/4.3，有两段休学经历，我应该如何写我的简历？

#### 让模型cosplay

BAD:帮我写一封投诉信投诉学校的保研政策

GOOD:假设你是一个在保研线边缘的学生，你本来手握着某学校的硕士offer。学校有一个读博士排名分数加五分的政策，很不幸你在保研的时候被后面的学生读博士加五分超过了，无法保研。你要如何写这封投诉信给教育局说明这种不公正的情况，争取回自己的保研名额。

#### 分割文本



用庄严诚恳的语气，帮我润色下面这篇申请书。


""" 

申请书内容

 """

### 提供参考文本

你是一个科技博主，我会在三重括号内给你提供示例。请你模仿示例，写出5个回答

提示词：称赞华为mate60

'''

提示词：称赞5G

不觉得这很酷吗？作为一名理工男我觉得这太酷了！很符合我对未来生活的想象，科技并带着趣味。

'''

### 将复杂任务拆分为更简单的子任务

### 给GPT时间思考

对于特定问题，让gpt自己先思考，再让其抉择对与错

### 使用外部工具


在下列三重括号内执行python代码，查看结果

'''

n=10

s=0

m='x'

for i in range(n):

    s += i

    m += 1

print(s,m)

'''

### 
