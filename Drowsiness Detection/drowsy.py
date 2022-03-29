import cv2
import dlib 
from scipy.spatial import distance
import winsound
from playsound import playsound

count = 0
def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks=dlib_facelandmark(gray,face)
        leftEye=[]
        rightEye=[]
        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        
        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        
        left_EAR = calculate_EAR(leftEye)
        right_EAR = calculate_EAR(rightEye)
        EAR = (left_EAR+right_EAR)/2
        EAR = round(EAR,2)
        if EAR<0.18:
            count = count + 1
            cv2.putText(frame,"DROWSY",(20,100),
        	cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
            cv2.putText(frame,"Are you sleepy?",(20,400),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
            
            if count > 2:
                
                winsound.PlaySound('Alarm.wav',winsound.SND_ASYNC)
                continue
            # playsound('Alarm.wav')
            print("Drowsy")
        print(EAR)
    cv2.imshow("Are you sleepy?",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
    