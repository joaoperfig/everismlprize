import image_slicer
from PIL import ImageDraw, ImageFont
from PIL import Image
import singlefruitnetwork

net = ournetwork.open_net("200_epochs.h5")


input_name = "apples"
output_name = input_name + "_out"
best_slices=0
best_avg = 0
slice_range = [9, 12, 16, 20, 25, 30, 36]

for num_slices in slice_range:  #identify best number of slices
    tiles = image_slicer.slice((input_name + '.jpg'), num_slices, save=False)
    
    image_slicer.save_tiles(tiles)  
    total= 0
    for tile in tiles:
        name = tile.filename  
        #print(name)
        tag, confidence = ournetwork.classify(net, name)
        #print(tag, confidence)
        total = total + confidence
    avg = total/num_slices
    if (avg >= best_avg):
        best_avg= avg
        best_slices = num_slices
    print("######################### ", num_slices, " -> ", avg)

tiles = image_slicer.slice((input_name + '.jpg'), best_slices, save=False)

image_slicer.save_tiles(tiles)  
total= 0
for tile in tiles: #slice and render bounding boxes
    name = tile.filename  
    print(name)
    tag, confidence = ournetwork.classify(net, name)
    print(tag, confidence)
    overlay = ImageDraw.Draw(tile.image)
    overlay.text((5, 5), str(tile.number), (125, 125, 125),
                 ImageFont.load_default())
    #overlay.text((5, 20), tag+" "+str(confidence) , (0, 0, 0),
    #             ImageFont.load_default())        
    overlay.text((5, 20), tag , (0, 0, 0),
                 ImageFont.load_default())            
    if(tag != "Other"):
        overlay.rectangle([0, 0, tile.image.width-5, tile.image.height-5], outline=5)



image = image_slicer.join(tiles)
image.save(output_name + '.jpg')  # save output