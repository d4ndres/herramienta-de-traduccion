
import drawOverScreen as dos
import ImageProcessor
import TextProcessor
import lookAt

class StreamCCApp:
	def __init__(self):
		self.image_processor = ImageProcessor.ImageProcessor()
		self.text_processor = TextProcessor.TextProcessor()
		self.selector_area_scream = dos.AreaSelector()
		self.coords = self.selector_area_scream.get_coordinates()
		self.prev_text = ''

	def action_get_text_from_image(self, image, translate = False):
		text = self.image_processor.image_to_string(image)
		if self.text_processor.similarity_ratio(text, self.prev_text) < 0.8:
			if translate:
				translated_text = self.text_processor.translateToEs(text)
				return translated_text.text
			else:
				return text
		self.prev_text = text

	def run(self):
		cropped_shadow = None

		textDisplay = lookAt.TextDisplay()
		while True:
			try:	
				image = self.image_processor.capture_screen(self.coords)
				areaCC = self.image_processor.find_black_area(image)
				cropped_image = self.image_processor.crop_image(image, areaCC)

				if not self.image_processor.equal_image(cropped_image, cropped_shadow):
					text = self.action_get_text_from_image(cropped_image, True)
					print(text)
					textDisplay.update_text(text)

				cropped_shadow = cropped_image
			
			# No areas found
			except AssertionError:
				pass

if __name__ == '__main__':
	app = StreamCCApp()
	app.run()