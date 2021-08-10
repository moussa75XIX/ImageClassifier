import os
from PIL import Image

espagne_folder = r"C:\Users\mouss\OneDrive\Documents\ProjetAnnuelRattrapage\dataset\espagne_to_resize"
france_folder = r"C:\Users\mouss\OneDrive\Documents\ProjetAnnuelRattrapage\dataset\france_to_resize"
japon_folder = r"C:\Users\mouss\OneDrive\Documents\ProjetAnnuelRattrapage\dataset\japon_to_resize"


espagne_final_folder = r"C:\Users\mouss\OneDrive\Documents\ProjetAnnuelRattrapage\dataset\espagne"
france_final_folder = r"C:\Users\mouss\OneDrive\Documents\ProjetAnnuelRattrapage\dataset\france"
japon_final_folder = r"C:\Users\mouss\OneDrive\Documents\ProjetAnnuelRattrapage\dataset\japon"


def resize(espagne_folder,espagne_final_folder,
           france_folder, france_final_folder,
           japon_folder,japon_final_folder ):

    for id,file in enumerate(os.listdir(espagne_folder)):
        file_path = os.path.join(espagne_folder, file)
        image = Image.open(file_path)
        image = image.resize((64, 64))
        image.save(r'{}\{}.png'.format(espagne_final_folder,str(id)), "PNG", quality=64)

    for id,file in enumerate(os.listdir(france_folder)):
        file_path = os.path.join(france_folder, file)
        image = Image.open(file_path)
        image = image.resize((64, 64))
        image.save(r'{}\{}.png'.format(france_final_folder,str(id+240)), "PNG", quality=64)

    for id,file in enumerate(os.listdir(japon_folder)):
        file_path = os.path.join(japon_folder, file)
        image = Image.open(file_path)
        image = image.resize((64, 64))
        image.save(r'{}\{}.png'.format(japon_final_folder,str(id+480)), "PNG", quality=64)


def main():
    resize(espagne_folder, espagne_final_folder,france_folder,france_final_folder,japon_folder,japon_final_folder)


if __name__ == "__main__":
    main()