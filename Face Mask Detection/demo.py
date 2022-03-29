import numpy as np
import cv2

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

with_mask = with_mask.reshape(200, 50 * 50 * 3)
without_mask = without_mask.reshape(200, 50 * 50 * 3)

print(with_mask.shape)
print(without_mask.shape)

X = np.r_[with_mask, without_mask]

print(X.shape)

labels = np.zeros(X.shape[0])

labels[200:] = 1.0

names = {0: 'Mask', 1: 'No mask'}

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.25)

print(X_train.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_train = pca.fit_transform(X_train)

print(X_train[0])

print(X_train.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.20)
svm = SVC()
svm.fit(X_train, Y_train)

# X_test = pca.transform(X_test)
y_pred = svm.predict(X_test)

print(accuracy_score(Y_test, y_pred))

haar_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)
            # face = pca.transform(face)
            pred = svm.predict(face)
            n = names[int(pred)]
            cv2.putText(img, n, (x, y), font, 1, (244, 250, 250), 2)
            print(n)

            if len(data) < 400:
                data.append(face)
        cv2.imshow("result", img)
        if cv2.waitKey(2) == 27:
            break

capture.release()
cv2.destroyAllWindows()
