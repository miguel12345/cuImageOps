import cv2
import numpy as np
from cuImageOps.operations import Affine

image = cv2.imread("/workspace/tests/data/grayscale_dog.jpg", cv2.IMREAD_UNCHANGED)

result = (
    Affine(translate=(10, 10), rotate=45, scale=(0.5, 1.0)).run(image).cpu().numpy()
)

cv2.imshow("Affine", result.astype(np.uint8))
cv2.waitKey()
