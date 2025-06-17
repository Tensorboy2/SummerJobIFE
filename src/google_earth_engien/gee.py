import ee
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Authenticate & initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='summer-job-ife')
def mask_s2_clouds(image):
    """Masks clouds in Sentinel-2 image using the QA60 band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 35
    cirrus_bit_mask = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask).divide(10000)

# Define location
lon, lat = 139.3767, 35.9839
region = ee.Geometry.BBox(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01)

# Load Sentinel-2 data
dataset = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region)
    .filterDate('2020-01-01', '2020-01-30')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(mask_s2_clouds)
)

# Average the collection and select RGB bands
image = dataset.mean().select(['B4', 'B3', 'B2'])

# Visualization parameters
vis_params = {
    'min': 0.0,
    'max': 0.3,
    'bands': ['B4', 'B3', 'B2'],
}

# Request thumbnail image
thumb_url = image.getThumbURL({
    'dimensions': 512,
    'region': region,
    'format': 'png',
    'min': vis_params['min'],
    'max': vis_params['max'],
    'bands': vis_params['bands']
})

# Download and show the image
response = requests.get(thumb_url)
img = Image.open(BytesIO(response.content))
image = np.array(img)
print(image.shape)
plt.imshow(image)
plt.axis('off')
plt.title("Sentinel-2 RGB (cloud-masked)")
plt.savefig("japan.jpg")
plt.show()
