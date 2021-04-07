from tensorflow.keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

#Загружаем модель (указывается путь не файлу модели, а к папке, содержащей saved_model.pb и variables)
model = load_model("model_folder/", compile = False)

#Указываем пути к изображениям, которые хотите сравнить
pathA = "test_images/1.png"
pathB = "test_images/2.png"

#Загружаем изображения в грейскейле (второй параметр - 0)
imageA = cv2.imread(pathA, 0)
imageB = cv2.imread(pathB, 0)
#Делаем ресайз (лучше в пейнте)
imageA = cv2.resize(imageA, (28,28), interpolation = cv2.INTER_AREA)
imageB = cv2.resize(imageB, (28,28), interpolation = cv2.INTER_AREA)
#Приводим в вид, необходимый для подачи в сеть (None,28,28,1).
imageA = np.expand_dims(imageA, axis=-1) #стало (28, 28, 1)
imageB = np.expand_dims(imageB, axis=-1)
imageA = np.expand_dims(imageA, axis=0) #стало (1, 28, 28, 1)
imageB = np.expand_dims(imageB, axis=0)
imageA = imageA / 255.0
imageB = imageB / 255.0
preds = model.predict([imageA, imageB])
diff = preds[0][0] #степень различия. 0 - одинаковые, 1 - разные.

print ("Difference equals ", diff)

#Для наглядности:
fig = plt.figure("Title", figsize=(4, 2))
plt.suptitle("Distance: {:.2f}".format(diff))
# Show the first image
ax = fig.add_subplot(1, 2, 1)
plt.imshow(np.squeeze(imageA), cmap=plt.cm.gray)
plt.axis("off")
# Show the second image
ax = fig.add_subplot(1, 2, 2)
plt.imshow(np.squeeze(imageB), cmap=plt.cm.gray)
plt.axis("off")
# Show the plot
plt.show()