import Augmentor
import os


def do_augmentation(input_dir="", output_dir="", max_files=0, max_folder=""):
    dirs = os.walk(input_dir)
    for s_dir in dirs:
        dir_path = s_dir[0]
        dir_files = s_dir[2]
        dir_name = os.path.basename(dir_path)

        print(f'Inside Folder: {dir_path}')
        print(f'Total Files: {len(dir_files)}')

        if not (len(dir_files) == 0 or dir_name == max_folder):
            new_dataset = os.path.join(output_dir, dir_name)
            p = Augmentor.Pipeline(source_directory=dir_path, output_directory=new_dataset)
            p.flip_left_right(probability=1)
            p.flip_top_bottom(probability=0.85)
            p.rotate(probability=0.90, max_left_rotation=25, max_right_rotation=25)
            p.zoom_random(probability=0.1, percentage_area=0.35)
            p.crop_centre(probability=0.85, percentage_area=0.75)
            p.crop_random(probability=0.75, percentage_area=0.65)
            p.resize(probability=1.0, width=256, height=256)
            p.sample(int((max_files / (len(dir_files))) * (len(dir_files))))
    return "All Gone Well"
