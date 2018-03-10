from PIL import Image
import sys
img0 = Image.open(sys.argv[1])
img1 = img0
pix = img1.load()
w, h = img1.size
for x in range(w):
	for y in range(h):
		(r, g, b) = pix[x,y]
		r = r//2
		g = g//2
		b = b//2
		pix[x,y] = (r,g,b)
img1.save("Q2.png", "PNG")