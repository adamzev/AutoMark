from extractor import Extractor
import helpers as Helpers

ext = Extractor('images/BCBA8F9752.jpg', False)

final = ext.final
#Helpers.show(final)

cells = Cells(final)
