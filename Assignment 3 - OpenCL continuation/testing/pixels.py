image = [
	[[0xff,0xff,0xff,0xff],  [0xff,0x00,0x00,0xff]],
	[[0x40,0x40,0x40,0xff],  [0x00,0xff,0x00,0xff]],
	[[0x00,0x00,0x00,0xff],  [0x00,0x00,0xff,0xff]]
]
imageGrey = [
	[0xff,  0x4c],
	[0x40,  0x96],
	[0x00,  0x1d]
]
def getPixel(x, y):
	try:
		return imageGrey[y][x]
	except IndexError:
		return 0x0

mask = [
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0
]
factor = 25.0


def filterPixel(pos):
	cx, cy = pos
	return (
		getPixel(cx-2,cy-2)*mask[ 0]/factor + getPixel(cx-2,cy-1)*mask[ 1]/factor + getPixel(cx-2,cy)*mask[ 2]/factor + getPixel(cx-2,cy+1)*mask[ 3]/factor + getPixel(cx-2,cy+2)*mask[ 4]/factor +
		getPixel(cx-1,cy-2)*mask[ 5]/factor + getPixel(cx-1,cy-1)*mask[ 6]/factor + getPixel(cx-1,cy)*mask[ 7]/factor + getPixel(cx-1,cy+1)*mask[ 8]/factor + getPixel(cx-1,cy+2)*mask[ 9]/factor +
		getPixel(cx  ,cy-2)*mask[10]/factor + getPixel(cx  ,cy-1)*mask[11]/factor + getPixel(cx  ,cy)*mask[12]/factor + getPixel(cx  ,cy+1)*mask[13]/factor + getPixel(cx  ,cy+2)*mask[14]/factor +
		getPixel(cx+1,cy-2)*mask[15]/factor + getPixel(cx+1,cy-1)*mask[16]/factor + getPixel(cx+1,cy)*mask[17]/factor + getPixel(cx+1,cy+1)*mask[18]/factor + getPixel(cx+1,cy+2)*mask[19]/factor +
		getPixel(cx+2,cy-2)*mask[20]/factor + getPixel(cx+2,cy-1)*mask[21]/factor + getPixel(cx+2,cy)*mask[22]/factor + getPixel(cx+2,cy+1)*mask[23]/factor + getPixel(cx+2,cy+2)*mask[24]/factor
	)


print( getPixel(1,0) )
print( int(filterPixel((0,0))), end=' ' )
print( int(filterPixel((1,0))))
print( int(filterPixel((0,0))), end=' ' )
print( int(filterPixel((1,1))))
print( int(filterPixel((0,1))), end=' ' )
print( int(filterPixel((1,1))))

