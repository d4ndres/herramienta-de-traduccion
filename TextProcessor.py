from googletrans import Translator

class TextProcessor:
  def __init__(self):
    self.translator = Translator()

  def translateToEs(self, text):
    return self.translator.translate( text , dest='es')