from images2video import ImagesToVideo
from images2video.effects import * # NOQA

video = ImagesToVideo(filename='test.avi', seconds=10)
video.add_image('model/test-T01/4.png', ResizeEffect)
video.add_image('model/test-T01/56.png', CropEffect, bounce=True)
video.add_image('model/test-T01/156.png', RotationEffect, bounce=True)
video.generate()