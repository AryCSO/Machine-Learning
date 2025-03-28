from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

melhor_k = 1
melhor_acuracia = 0

for k in range(1, 11):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_temp, X_train, y_train, cv=5)
    media_acuracia = scores.mean()
    
    print(f"K={k}, Acurácia média: {media_acuracia:.4f}")
    
    if media_acuracia > melhor_acuracia:
        melhor_acuracia = media_acuracia
        melhor_k = k

print(f"\nMelhor K encontrado: {melhor_k}, Acurácia média: {melhor_acuracia:.4f}")

knn = KNeighborsClassifier(n_neighbors=melhor_k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo final:", accuracy)
