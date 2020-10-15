import matplotlib.pyplot as plt


labels = ['First Slice', 'Middle Slice']
men_means = [20, 35]
women_means = [25, 32]
men_std = [2, 3]
women_std = [3, 5]
width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, men_means, width, yerr=men_std, label='Men')
ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
       label='Women')

ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.legend()

plt.show()
