import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# #########################################
# 임의의 데이터 생성
x = np.linspace(0, np.pi, 100)
y = np.linspace(0, np.pi, 100)
x, y = np.meshgrid(x, y)

# z = np.sin(np.sqrt(x**2 + y**2))
z = 1 / (50 * x / np.pi + 2) + 1 / 2 + np.minimum((np.arctanh(1. - np.maximum(2 * y / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5

# 3차원 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3차원 그래프에 데이터 플로팅
ax.plot_surface(x, y, z)
ax.set_xlabel("AO")
ax.set_ylabel("TA")
ax.set_zlabel("Reward")

plt.show()
#########################################

# x = np.linspace(0, 100, 100)
# y = 1 * (x < 5) + (x >= 5) * np.clip(-0.032 * x**2 + 0.284 * x + 0.38, 0, 1) + np.clip(np.exp(-0.16 * x), 0, 0.2)

# # 그래프를 그립니다.
# plt.plot(x, y)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Range Reward')
# plt.grid(True)

# # 그래프 표시
# plt.show()