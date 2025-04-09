# alambition
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# تحميل البيانات
data = pd.read_csv('data.csv')

# إزالة القيم المفقودة
data.dropna(inplace=True)

# تحويل النصوص إلى حروف صغيرة
data['description'] = data['description'].str.lower()

# إزالة الكلمات غير المفيدة (stop words)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['description'] = data['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# تحويل النصوص إلى تمثيل رقمي باستخدام TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['description'])
import cv2

# قراءة صورة
image = cv2.imread('image.jpg')

# تحسين الجودة (تصفية الضوضاء)
denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# تحويل الصورة إلى تدرج الرمادي
gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

# حفظ الصورة المعالجة
cv2.imwrite('processed_image.jpg', gray_image)
import cv2

# فتح الفيديو
video_capture = cv2.VideoCapture('video.mp4')

# استخراج الإطارات من الفيديو
frame_count = 0
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    frame_count += 1
    # حفظ كل 10 إطارات كمثال
    if frame_count % 10 == 0:
        cv2.imwrite(f'frame_{frame_count}.jpg', frame)

video_capture.release()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.3, random_state=42)

# تدريب نموذج RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# التنبؤ بالبيانات
y_pred = clf.predict(X_test)

# قياس دقة النموذج
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
import tensorflow as tf
from tensorflow.keras import layers, models

# بناء نموذج بسيط للشبكة العصبية
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # بالنسبة لمشكلة تصنيف ثنائي
])

# تجميع النموذج
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(X_train, y_train, epochs=10, batch_size=32)

# تقييم النموذج
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# حساب المصفوفة
cm = confusion_matrix(y_test, y_pred)

# عرض المصفوفة باستخدام Seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
from flask import Flask, request, jsonify
import numpy as np

app = Flask(_name_)

# تحميل النموذج المدرب
model = tf.keras.models.load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # الحصول على البيانات من الطلب
    features = np.array(data['features']).reshape(1, -1)
    
    # التنبؤ باستخدام النموذج
    prediction = model.predict(features)
    
    # إرجاع النتيجة
    return jsonify({'prediction': prediction[0][0]})

if _name_ == '_main_':
    app.run(debug=True)
    
