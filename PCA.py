import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



# 예제 데이터 생성
np.random.seed(0)
X = np.random.randn(100,2)
X[:,1] = 0.8 * X[:,0] + 0.4 * np.random.randn(100)

# PCA 적용 (2차원에서 1차원으로 축소)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# 축소된 데이터를 원래 차원으로 복원
X_restored = pca.inverse_transform(X_pca)

# 결과 시각화
plt.scatter(X[:,0], X[:,1], alpha=0.7, label="Original Data")
plt.scatter(X_restored[:,0], X_restored[:,1], alpha=0.7, label="Restored Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.title("PCA Example")
st.pyplot(plt)