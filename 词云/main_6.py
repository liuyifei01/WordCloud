from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import jieba.analyse
from PIL import Image
import numpy as np

# 词云精细控制：控制每个词的大小，使用generate_from_frequencies()
# generate_from_frequencies()包括两个参数
#   frequencies：一个字典，用于指定词对应的大小
#   max_font_size：最大字号，一般为None
# generate() = process_text() + generate_from_frequencies()

# 打开文本
text = open('xyj.txt', encoding='utf-8').read()

# 提取关键词和权重
freq = jieba.analyse.extract_tags(text, topK=200, withWeight=True)  # 列表，该列表的0为词，1为权重
print(freq[:20])
freq = {i[0]: i[1] for i in freq}   # 将列表转成字典

# 生成对象
stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]
mask = np.array(Image.open('color_mask.png'))
wc = WordCloud(
    mask=mask,
    font_path='Hiragino.ttf',
    background_color=None,
    width=800,
    height=600,
    mode='RGBA',
    stopwords=stopwords,
).generate_from_frequencies(freq)

# 从图片中生成颜色
image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file('xyj_main_6.png')
