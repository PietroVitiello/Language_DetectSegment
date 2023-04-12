from PIL import Image, ImageDraw

def open_image(path):
    return Image.open(path)

def draw_bounding_boxes(image, bbox):
    bbox = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    img1 = ImageDraw.Draw(image)  
    img1.rectangle(bbox, outline ="red", width=4)