import cv2
import matplotlib.pyplot as plt

img = cv2.imread('transform.jpg')

# vamos ha volver los blanco mas blancos
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#bitwise mask
mask = cv2.inRange(gray, 80, 255)
result = cv2.bitwise_and(gray, gray, mask=mask)


plt.imshow(result, cmap='gray')
plt.show()

#cv2.imshow('Result', result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
