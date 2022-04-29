img = cv2.imread("cameraman.png", 0)
img = cv2.resize(img, (128, 128))
[nrow, ncol] = img.shape

[xCoor, yCoor] = np.mgrid[0:nrow, 0:ncol]
pt.figure()

ax = pt.axes(projection="3d")
ax.plot_surface(xCoor, yCoor, img, cmap=pt.cm.jet)

pt.show()
