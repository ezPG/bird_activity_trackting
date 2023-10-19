import deeplabcut


# ProjectFolderName = './birds'
# VideoType = 'mp4' 

# #don't edit these:
# videofile_path = [ProjectFolderName+'/videos/'] #Enter the list of videos or folder to analyze.
# print(f"[Info] video file path: {videofile_path}")


path_config_file = "./birds/config.yaml"

deeplabcut.load_demo_data(path_config_file)

deeplabcut.train_network(path_config_file, shuffle=1, displayiters=100,saveiters=500, maxiters=10000)

deeplabcut.evaluate_network(path_config_file,plotting=True)