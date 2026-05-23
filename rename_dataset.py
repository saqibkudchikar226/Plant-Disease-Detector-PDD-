import os

dataset_path = "dataset/train"

for root, dirs, files in os.walk(dataset_path):

    for count, file in enumerate(files):

        old_path = os.path.join(root, file)

        if not os.path.exists(old_path):
            continue

        ext = os.path.splitext(file)[1]

        new_name = f"img_{count}{ext}"

        new_path = os.path.join(root, new_name)

        try:
            os.rename(old_path, new_path)
            print("Renamed:", old_path)

        except Exception as e:
            print("Skipped:", old_path)
            print(e)

print("DONE")