import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 字体管理器

x_data = ['1', '5', '10', '20', '30', '40', '50']
y_data = [0.65,0.82,0.86,0.89,0.91,0.92,0.93]
y_data2 = [0.46,0.65,0.72,0.78,0.8,0.83,0.84]
y_data3 =[0.52,0.68,0.73,0.76,0.78,0.79,0.79]
y_data4 =[0.34,0.54,0.62,0.69,0.73,0.75,0.78]
y_data5=[0.38,0.59,0.67,0.74,0.78,0.80,0.82]


ln1, = plt.plot(x_data, y_data, color='red', linewidth=2.0, linestyle='-')
ln2, = plt.plot(x_data, y_data2, color='blue', linewidth=3.0, linestyle='--')
ln3, = plt.plot(x_data, y_data3, color='yellow', linewidth=3.0, linestyle='-')
ln4, = plt.plot(x_data, y_data4, color='black', linewidth=3.0, linestyle='-')
ln5, = plt.plot(x_data, y_data5, color='cyan', linewidth=3.0, linestyle='-')
my_font = fm.FontProperties(fname="C:/Windows/Fonts/simsun.ttc")

#plt.title("电子产品销售量", fontproperties=my_font)  # 设置标题及字体

plt.legend(handles=[ln1, ln2,ln3,ln4,ln5], labels=['IAPM+', 'IAPM','FashionNet','DenseNet','SiameseNet'], prop=my_font)


ax = plt.gca()
ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示

plt.xlabel('Top k')
plt.ylabel('Accuracy')
plt.savefig("examples.jpg",dpi=600)
plt.show()
