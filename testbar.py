from matplotlib import pyplot as plt

x = range(1,11)
x = ["herman","kiko"]
y = [84,87]
fig, ax = plt.subplots()
# 截尾平均数

b = ax.bar(x, y)
plt.title('Recommended song list score')
for a, b in zip(x, y):
    ax.text(a, b+1, b, ha='center', va='bottom')

plt.xlim((-1,10))

plt.ylim((0,100))
plt.xticks(range(len(x)+2))
plt.xlabel('playlist number')
plt.ylabel('score')
plt.legend()
plt.show()