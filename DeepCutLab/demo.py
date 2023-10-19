import deeplabcut

ProjectFolderName = 'birds'
VideoType = 'mp4' 
path_config_file = ProjectFolderName+'/config.yaml'

videofile_path = ['vid1.mp4','vid2.mp4','vid3.mp4']
deeplabcut.analyze_videos(path_config_file,"vid_examples/"+videofile_path, videotype='.mp4')
deeplabcut.create_labeled_video(path_config_file,"vid_examples/"+videofile_path)
deeplabcut.plot_trajectories(path_config_file,"vid_examples/"+videofile_path)