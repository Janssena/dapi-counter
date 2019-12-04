import os
import glob
import helpers
from PIL import Image
from countingwindow import CountingWindow
from definitions import PATH

files = sorted(glob.glob(PATH["IMAGE_IN"] + "*.png"))
processed = glob.glob(PATH["IMAGE_OUT"] + "*.png")


print('Current progress is {}%'.format((len(processed) / len(files)) * 100))

for file in files:
    if helpers.is_processed(file, processed):
        continue

    file_properties = helpers.get_properties_from_filename(file)
    img = Image.open(file)
    file_properties["original_size"] = img.size

    # crop the image in one smaller image of 256x256
    crop_dimensions = file_properties["crop"]
    img_crop = img.crop(crop_dimensions)

    file_properties["name"] = '{}-{}-[{}].png'.format(
        file_properties["id"],
        file_properties["crop"],
        str(file_properties["slice"])
    )

    cw = CountingWindow(img_crop)
    cw.open_window()
    file_properties["count"], file_properties["locations"] = cw.get_results()
    if file_properties["count"] is not None and file_properties["locations"] is not None:
        helpers.write_to_csv(file_properties, PATH["CSV"])
        img_crop.save(PATH["IMAGE_OUT"] + file_properties["name"])
