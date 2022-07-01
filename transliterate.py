#import codecs,string
#from indic_transliteration import sanscript
#from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
from googletrans import Translator

def op_bhai(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text


'''def is_hindi(character):
    maxchar = max(character)
    if u'\u0900' <= maxchar <= u'\u097f':
        print(character)
        return character
    else:
        print(transliterate(character, sanscript.ITRANS, sanscript.DEVANAGARI))
        return transliterate(character, sanscript.ITRANS, sanscript.DEVANAGARI)'''

#character = op_bhai("ladkiyon ko khana banana chahiye")
#print("TRANSLATED : "+character)

