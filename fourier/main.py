import helper
import matplotlib.pyplot as plt
import cv2
plt.rc('text', usetex=True)

folder, num_bins = helper.arguments()
images = helper.scan_folder(folder)
if len(images) < 1:
    print('no images found in %r' % folder)
print(u"""------------------------------------------------------------------------------
OBS:: patches have been inverted to white (255), while interpatches were inverted to black (0).
With this change, the fourier transform refers to patches, not interpatches
------------------------------------------------------------------------------""")
for image,name in images:
    print(name)
    #
    # patches should be white and non-patches should be black
    #
    new_image = cv2.bitwise_not(image)
    helper.generate_graphs(new_image,name,plt)
