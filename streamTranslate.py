import cv2
import difflib
import drawOverScreen as dos
import ImageProcessor
import TextProcessor

def similarity_ratio(text1, text2):
    return difflib.SequenceMatcher(None, text1, text2).ratio()


class StreamCCApp:
	def __init__(self):
		self.image_processor = ImageProcessor.ImageProcessor()
		self.text_processor = TextProcessor.TextProcessor()
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