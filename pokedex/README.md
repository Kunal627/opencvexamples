# Pokedex from pyimagesearch blogs by Adrian Rosebrock
https://www.pyimagesearch.com/2014/04/07/building-pokedex-python-indexing-sprites-using-shape-descriptors-step-3-6/


scrapepokemon.py - web scrapper.
--------------------------------
scrapes pokemon sprites from https://pokemondb.net
parse all links from source of https://pokemondb.net using Beautifulsoup

python .\scrapepokemon.py --htmlPath ..\data\inp\pokedex.html --outPath ..\data\inp\pokemon\


index.py -- index images on Zernikemoments
--------
python .\index.py --sprites ..\data\inp\pokemon\ --index ..\data\out\index.cpickle

find_screen.py - crop the pokemon from gameboy screen (saves the cropped_image.png to /data/test folder)
-------------
python .\find_screen.py --query ..\data\test\gameboy.png

search
------
python search.py --index ..\data\out\index.cpickle --query ..\data\test\pikachu.png

NB- My code implementation of pokedex is broken does not return correct image as output.  may be its the input images not indexed correctly. Actual blog post only indexed Red blue. 
