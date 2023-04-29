import cv2

alpha = 1.0 
beta = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    cv2.imshow('Gray Video', gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
