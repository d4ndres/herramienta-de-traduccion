import cv2
import numpy as np
from PIL import ImageGrab
import pytesseract as tst
tst.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


class ImageProcessor:
    def capture_screen(self, coords=None):
        captured 	= ImageGrab.grab()
        capturer_np = np.array(captured)
        image 		= cv2.cvtColor( capturer_np, cv2.COLOR_RGB2BGR)
        if coords is not None:
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            image = image[y1:y2, x1:x2]
        return image	

    def find_black_area(self, image) -> tuple:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(image_gray, 0, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) > 0, 'No areas found'
        maxAreaContour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(maxAreaContour)

    def crop_image(self, image, bounding_rect):
        x, y, w, h = bounding_rect
        return image[y:y+h, x:x+w]
    
    def image_to_string(self, image):
        return tst.image_to_string(image)
	
    def equal_image(self, image, image_shadow):
        return np.array_equal(image, image_shadow)