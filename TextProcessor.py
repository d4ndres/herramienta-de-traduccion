from googletrans import Translator
import difflib

class TextProcessor:
  def __init__(self):
    self.translator = Translator()

  def translateToEs(self, text):
    try:
      return self.translator.translate( text , dest='es')
    except:
      return text
    
  def similarity_ratio(self, text1, text2):
    return difflib.SequenceMatcher(None, text1, text2).ratio()