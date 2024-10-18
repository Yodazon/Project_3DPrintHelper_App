#A script to sort photos into proper folders based on data in a CSV
#Class Labels Are 
# 0: Good
# 1: Under-Extrusion
# 2: Stringing
# 4: Spaghetti
#each file in column "image" of the all_images_no_filter.csv has a class in the second column based on the following labels
#goal is to automatically sort images into folders
#March 5, 2024

import pandas as pd
import os
from PIL import Image

def make_newFolder(folderName):
    path = f'images\\{folderName}'

    try:
        os.makedirs(path)
    except OSError as error:
        print(error)


def sort_photos(folderName, fileName):
    path = f'images\\{folderName}\\{fileName}'
    img_filePath = Image.open(f"Printing_Errors\\images\\all_images256\\{fileName}")

    img_filePath.save(path)



my_class = { '0':'good','1':'underextrusion', '2':'stringing', '4':'spaghetti'}

img_dataset = pd.read_csv("Printing_Errors\\general_data\\all_images_no_filter.csv")

#For testing, to print out a specific row
#print(img_dataset[img_dataset['image'] == 'ELP_12MP_01.12.2022_166990882982.png'])

for value in my_class.values():
    make_newFolder(value)

for index, row in img_dataset.iterrows():
    file_name = row.iloc[0]
    class_value = str(row.iloc[1])

    sort_photos(my_class[class_value], file_name)

