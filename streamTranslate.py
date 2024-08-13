from PIL import ImageGrab
import cv2
import numpy as np
import pytesseract as tst
from googletrans import Translator
tst.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def captureMainScreen( save = False ):
	captured 	= ImageGrab.grab()
	capturer_np = np.array(captured)
	image 		= cv2.cvtColor( capturer_np, cv2.COLOR_RGB2BGR)
	imageGray 		= cv2.cvtColor( capturer_np, cv2.COLOR_RGB2GRAY)
	if save:
		cv2.imwrite('prueba.png', image)
	return image, imageGray

def readImage( path ):
	image = cv2.imread( path )
	imageGray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
	return image, imageGray

def getCloseCaptions( image, imageGray ):
	# encuentra areas de con valor de pixel 0
	mask = cv2.inRange(imageGray, 0, 0)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	maxAreaContour = max(contours, key=cv2.contourArea)
	# Find the bounding rectangle of the contour
	x, y, w, h = cv2.boundingRect(maxAreaContour)

	# Crop the image using the bounding rectangle coordinates
	cropped_image = image[y:y+h, x:x+w]
	return cropped_image

if __name__ == '__main__':
	translator = Translator()
	cropped_shadow = None
	five_previous_text = []

	while True:
		image, imageGray = captureMainScreen()
		cropped_image = getCloseCaptions( image, imageGray )

		if cropped_shadow is not None:
			if not np.array_equal(cropped_image, cropped_shadow):
				text = tst.image_to_string(cropped_image)
				if text not in five_previous_text:
					five_previous_text.append( text )
					res = translator.translate( text , dest='es')
					print( res.text, len(text) )
				
				if len(five_previous_text) > 5:
					five_previous_text.pop(0)

				


		cropped_shadow = cropped_image

	cv2.destroyAllWindows()