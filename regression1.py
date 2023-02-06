from sklearn.datasets import load_breast_cancer
import introduction_to_ml_with_python.mglearn as mglearn
import matplotlib.pyplot as plt
import mplcyberpunk


X, y = mglearn.datasets.make_wave(n_samples=40)

# cancer = load_breast_cancer()
# print(cancer.data.shape)

# plt.figure(figsize=(12, 7))
# plt.subplot(2, 2, 1)
mglearn.plots.plot_knn_classification(n_neighbors=1)

# plt.subplot(2, 2, 4)
# # plt.style.use('cyberpunk')
# plt.ylim(-3, 3)
# plt.xlabel('Feature')
# plt.ylabel('Target variable')
# plt.plot(X, y, 'o')
plt.show()
