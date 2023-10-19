import cv2
import torch
import torchvision
import numpy as np
import argparse
from model import get_model
from matplotlib import pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to the input video")
ap.add_argument("-o", "--output", required=True, help="path to the output video")
args = vars(ap.parse_args())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load the model
model = get_model().to(device)
model.load_state_dict(torch.load('faster_rcnn_model.pth'))
model.eval()

# set up the video capture
print(f"[INFO] loading video {args['video']}...")
print(f"[INFO] saving video to {args['output']}...")

cap = cv2.VideoCapture(args['video'])

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = f"{args['video'].split('/')[-1].split('.')[0]}_model_output"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

print("[INFO] processing video...")
# print(f"[INFO] press 'q' to stop the video...")
i=0
# loop through the frames of the video
while True:
    # read in the next frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = np.transpose(frame, (2, 0, 1))
    # img = torch.from_numpy(frame)
    # img = img.float().to(device)
    # # convert the frame to a PyTorch tensor
    img = torch.from_numpy(frame[:, :, [2, 1, 0]])
    plt.imsave(f"outputs/{save_name}_{i}.png",img.cpu().numpy().astype(np.uint8))
    i+=1
    img = img.permute(2,0,1).unsqueeze(0).float().to(device)

    # make the prediction
    with torch.inference_mode():
        pred = model(img)
    
    # print(f"pred: {pred}")
    # extract the predicted bounding boxes and labels
    # Get predicted bounding boxes and object classes
    
    boxes = pred[0]['boxes'].cpu().numpy()
    labels = pred[0]['labels'].cpu().numpy()

    # loop through the predicted boxes and draw them on the frame
    # for i in range(len(boxes)):
    box = boxes[0]
    label = labels[0]
    x1, y1, x2, y2 = box.astype(np.int32)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # write the frame to the output video
    # print(f"[INFO] writing frame {int(cap.get(1))} / {int(cap.get(7))}")
    # frame = np.transpose(frame, (1, 2, 0))
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # plt.imsave(f"outputs/{save_name}_{i}.png",frame.cpu().numpy().astype(np.uint8))
    plt.imsave(f"outputs/{save_name}_{i}.png",frame)
    out.write(frame)

    # display the frame
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# release the video capture and close the window
out.release()
cap.release()
print(f"[INFO] video saved to {args['output']}")