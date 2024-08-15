import cv2
import difflib
import drawOverScreen as dos
from googletrans import Translator
import numpy as np
from PIL import ImageGrab
import pytesseract as tst
tst.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def similarity_ratio(text1, text2):
		return difflib.SequenceMatcher(None, text1, text2).ratio()

class TextProcessor:
	def __init__(self):
		self.translator = Translator()

	def translateToEs(self, text):
		return self.translator.translate( text , dest='es')





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

class StreamCCApp:
	def __init__(self):
		self.image_processor = ImageProcessor()
		self.text_processor = TextProcessor()
		self.selector_area_scream = dos.AreaSelector()
		self.coords = self.selector_area_scream.get_coordinates()

	def run(self):
		cropped_shadow = None
		previous_text = ''
		while True:
			try:	
				image = self.image_processor.capture_screen(self.coords)
				areaCC = self.image_processor.find_black_area(image)
				cropped_image = self.image_processor.crop_image(image, areaCC)

				if cropped_shadow is not None and not self.image_processor.equal_image(cropped_image, cropped_shadow):
					text = self.image_processor.image_to_string(cropped_image)
					if similarity_ratio(text, previous_text) < 0.8:
						translated_text = self.text_processor.translateToEs(text)
						print(translated_text.text)
					previous_text = text
				elif cropped_shadow is None:
					text = self.image_processor.image_to_string(cropped_image)
					translated_text = self.text_processor.translateToEs(text)
					print(translated_text.text)
					previous_text = text


				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

				cropped_shadow = cropped_image
			
			# No areas found
			except AssertionError:
				pass

if __name__ == '__main__':
	app = StreamCCApp()
	app.run()