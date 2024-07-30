import cv2

cat_face_cascade=cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
human_face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,img=cap.read()
    if not ret:
        break

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    catfaces=cat_face_cascade.detectMultiScale(gray,1.1,4)
    faces=human_face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in catfaces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        label=(x,y-10)
        cv2.putText(img,"cat",label,cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        label=(x,y-10)
        cv2.putText(img,"human",label,cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
