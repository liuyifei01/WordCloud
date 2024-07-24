from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
import numpy as np
from PIL import Image
import random

# 将词云的颜色改为统一颜色，只改变亮度和渲染

# 打开文本
text = open('xyj.txt', encoding='utf-8').read()

# 中文分词
text = ' '.join(jieba.cut(text))


# 定义一个颜色函数
def random_color(word, font_size, position, orientation, font_path, random_state):
    s = 'hsl(0, %d%%, %d%%)' % (random.randint(60, 80), random.randint(60, 80))
    print(s)
    return s


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
    color_func=random_color,
).generate(text)

# # 从图片中生成颜色
# image_colors = ImageColorGenerator(mask)
# wc.recolor(color_func=image_colors)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file('xyj_main_5.png')
