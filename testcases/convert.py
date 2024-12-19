from PIL import Image
import sys

if len(sys.argv) != 3:
    print('[Usage] python convert.py input.png output.png')
    sys.exit()

image = Image.open(sys.argv[1])
rgb_image = image.convert('RGB')
rgb_image.save(sys.argv[2])
