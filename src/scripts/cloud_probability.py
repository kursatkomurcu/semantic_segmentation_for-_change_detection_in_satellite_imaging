import ee

# Earth Engine'i başlat
ee.Initialize()

# Bir görüntü koleksiyonu tanımla ve tarih aralığına göre filtrele
collection = ee.ImageCollection('LANDSAT/LC08/C01/T1') \
    .filterDate('2020-01-01', '2020-12-31') \
    .filterBounds(ee.Geometry.Point(28.9784, 41.0082)) \
    .sort('CLOUD_COVER')

# Koleksiyondan en az bulutlu görüntüyü seç
image = collection.first()

# Görüntü bilgilerini yazdır
print(image.getInfo())
