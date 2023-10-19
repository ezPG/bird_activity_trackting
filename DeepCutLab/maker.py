import deeplabcut

task='Pose' # Enter the name of your experiment Task
experimenter='d22cs051' # Enter the name of the experimenter
video=['vid_examples/vid1.mp4','vid_examples/vid2.mp4'] # Enter the paths of your videos OR FOLDER you want to grab frames from.


path_config_file=deeplabcut.create_new_project(task,experimenter,video,copy_videos=True) 

exrtacted_frames = deeplabcut.extract_frames?

exrtacted_frames = deeplabcut.extract_frames(path_config_file) 

deeplabcut.extract_frames(path_config_file,'manual')

deeplabcut.check_labels(path_config_file) #this creates a subdirectory with the frames + your labels

deeplabcut.create_training_dataset(path_config_file)


deeplabcut.train_network(path_config_file)

deeplabcut.evaluate_network(path_config_file, plotting=True)

