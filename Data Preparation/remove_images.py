import os

# script for removing the images that I copied their paths manually
counter = 0
with open('to_remove.txt') as f:

    while True:
        file_path = f.readline()
        if not file_path:
            break

        file_path = file_path.strip()

        if os.path.isfile(file_path):
            os.remove(file_path)
            counter += 1

print(f"Counter {counter}")
