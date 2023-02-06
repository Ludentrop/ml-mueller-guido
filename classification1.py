import introduction_to_ml_with_python.mglearn as mglearn
import matplotlib.pyplot as plt


X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['Class 0', 'Class 1'], loc=4)
plt.xlabel('First feature')
plt.ylabel('Second feature')
print(f'X massive shape: {X.shape}')
plt.show()
