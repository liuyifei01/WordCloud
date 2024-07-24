from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
import numpy as np
from PIL import Image

# 将词云的形状与mask相同

# 打开文本
text = open('xyj.txt', encoding='utf-8').read()

# 中文分词
text = ' '.join(jieba.cut(text))
print(text[:100])

# 生成对象
mask = np.array(Image.open("black_mask.png"))
wc = WordCloud(mask=mask, background_color=None, mode='RGBA', font_path='Hiragino.ttf').generate(text)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file('xyj_mask.png')
