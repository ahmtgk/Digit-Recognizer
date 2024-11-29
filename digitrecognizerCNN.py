import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Veri setini yükleme
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Eğitim verilerini ayırma
X = train_data.iloc[:, 1:].values  # Pixel değerleri
y = train_data.iloc[:, 0].values  # Etiketler (0-9)

# Normalize et (0-255 arasındaki değerleri 0-1 arasına getir)
X = X / 255.0
test_data = test_data / 255.0

# Görüntü boyutlarını CNN için yeniden şekillendir
X = X.reshape(-1, 28, 28, 1)  # 28x28x1 (gri tonlamalı görüntü)
test_data = test_data.values.reshape(-1, 28, 28, 1)

# Etiketleri One-Hot Encoding ile dönüştür
y = to_categorical(y)

# Eğitim ve doğrulama veri setlerini ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN modelini oluştur
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Çıkış katmanı (10 sınıf)
])

# Modeli derle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Eğitim ve doğrulama doğruluğunu çizdir
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.title('Model Doğruluğu')
plt.show()

# Test veri seti için tahminler
predictions = model.predict(test_data)

# Tahminleri en olası sınıfa çevir
predicted_classes = np.argmax(predictions, axis=1)

# İlk 10 tahmini görselleştir
for i in range(10):
    plt.imshow(test_data[i].reshape(28, 28), cmap='gray')
    plt.title(f"Model Tahmini: {predicted_classes[i]}")
    plt.axis('off')
    plt.show()