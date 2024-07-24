from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 英文词云

# 打开txt文本
text = open('constitution.txt', encoding='utf-8').read()
# 生成对象
wc = WordCloud().generate(text)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file('wordcloud.png')