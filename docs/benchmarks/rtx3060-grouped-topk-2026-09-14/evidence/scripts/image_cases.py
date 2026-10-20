"""Deterministic image regressions, not a labeled accuracy dataset."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent / 'image_cases'
out.mkdir(exist_ok=False)
cv2.setNumThreads(1)
records = []
def save(name, image, recipe):
    path = out / (name + '.png')
    if not cv2.imwrite(str(path), image):
        raise RuntimeError('Cannot save image')
    records.append({'file':path.name, 'shape':list(image.shape), 'recipe':recipe,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})

for name in ['bus', 'zidane']:
    image = cv2.imread(str(root / 'images' / (name + '.jpg')))
    if image is None:
        raise RuntimeError('Cannot read source')
    h,w = image.shape[:2]
    save(name + '_original', image, 'Original JPEG decoded with cv2 ' + cv2.__version__)
    save(name + '_flip', image[:, ::-1], 'Horizontal mirror')
    save(name + '_half', cv2.resize(image, (w//2,h//2)), 'Half-size INTER_LINEAR')
    save(name + '_large', cv2.resize(image, (w*3//2,h*3//2)), '1.5-size INTER_LINEAR')
    save(name + '_dark', np.clip(image.astype(np.float32)*.35,0,255).astype(np.uint8), 'Brightness times0.35')
    save(name + '_bright', np.clip(image.astype(np.float32)*1.6,0,255).astype(np.uint8), 'Brightness times1.6')
    save(name + '_crop', image[h//5:h*4//5,w//5:w*4//5], 'Center60percent crop')
    small = cv2.resize(image,(320,320))
    save(name + '_tiles', np.tile(small,(2,2,1)), 'Four320squaretiles; dense synthetic case')
for value in [0,114,255]:
    save('blank_' + str(value), np.full((479,641,3),value,dtype=np.uint8), 'Constant BGR')
rng = np.random.default_rng(260013)
save('noise',rng.integers(0,256,size=(63,81,3),dtype=np.uint8),'Seed260013 integer noise')
board = ((np.indices((19,31)).sum(axis=0)%2)*255).astype(np.uint8)
save('checker',np.repeat(board[:,:,None],3,axis=2),'19x31 alternating checker')
(out / 'metadata.json').write_text(json.dumps(records,indent=2)+'\n')
print(str(len(records)) + ' image fixtures prepared')
