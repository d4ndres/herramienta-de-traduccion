from PIL import ImageGrab
import cv2
import numpy as np
import pytesseract as tst
from googletrans import Translator
tst.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import drawOverScreen as dos
import difflib

def similarity_ratio(text1, text2):
    return difflib.SequenceMatcher(None, text1, text2).ratio()

def captureMainScreen( coords = None, save = False ):
	captured 	= ImageGrab.grab()
	capturer_np = np.array(captured)
	image 		= cv2.cvtColor( capturer_np, cv2.COLOR_RGB2BGR)
	imageGray 		= cv2.cvtColor( capturer_np, cv2.COLOR_RGB2GRAY)

	if coords is not None:
		x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
		image = image[y1:y2, x1:x2]
		imageGray = imageGray[y1:y2, x1:x2]

	if save:
		cv2.imwrite('prueba.png', image)
	return image, imageGray

def readImage( path ):
	image = cv2.imread( path )
	imageGray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
	return image, imageGray

def getCloseCaptions( image, imageGray ):
	# Find areas with pixel value of 0
	mask = cv2.inRange(imageGray, 0, 0)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	if len(contours) == 0:
		return None 
	
	maxAreaContour = max(contours, key=cv2.contourArea)
	# Find the bounding rectangle of the contour
	x, y, w, h = cv2.boundingRect(maxAreaContour)

	# Crop the image using the bounding rectangle coordinates
	cropped_image = image[y:y+h, x:x+w]
	return cropped_image

if __name__ == '__main__':
	selector = dos.AreaSelector()
	coords = selector.get_coordinates()
	image, imageGray = captureMainScreen( coords )
	
	translator = Translator()
	cropped_shadow = None
	previous_text = ""

	mask = cv2.inRange(imageGray, 0, 0)

	while True:
		image, imageGray = captureMainScreen( coords )
		cropped_image = getCloseCaptions( image, imageGray )
		cv2.imshow('Captura', image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		if cropped_shadow is not None and cropped_image is not None:
			if not np.array_equal(cropped_image, cropped_shadow):
				text = tst.image_to_string(cropped_image)
				if similarity_ratio(text, previous_text) < 0.8:
					try:
						res = translator.translate( text , dest='es')
						print( res.text )
						previous_text = text
					except Exception as e:
						print(e)
					
		cropped_shadow = cropped_image

	cv2.destroyAllWindows()