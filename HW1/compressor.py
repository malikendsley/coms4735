import os


# compress every image in the library to 1000x1000 and save back over it
for root, _, files in os.walk(os.path.join(os.getcwd(), 'library\in')):
        for file in files:
            if file.endswith('.jpeg'):
            