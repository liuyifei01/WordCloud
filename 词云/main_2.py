from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba

# 中文词云

# 打开txt文本
text = open('xyj.txt', encoding='utf-8').read()

# 中文需要中文分词
text = ' '.join(jieba.cut(text))     # 返回一个与列表相似的结构
print(text[:100])

# 生成对象
wc = WordCloud(font_path='Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file('xyj.png')