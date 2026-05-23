import os
from PIL import Image

dataset_path = "dataset/train"

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        path = os.path.join(root,file)

        if not os.path.exists(path):
            continue

        try:
            img = Image.open(path)
            img.verify()

        except Exception as e:

            print("Bad image:",path)

            try:
                os.remove(path)
                print("Deleted.")
            except:
                print("Cannot delete.")

print("Cleaning finished.")