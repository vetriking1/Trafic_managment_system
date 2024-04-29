
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials , db
# load model and load the video
model = YOLO("./models/yolov8m.pt")
cap = cv2.VideoCapture("./videos/video1.mp4")

def train_model(x,y):
    x = np.array([int(day[3:]) for day in days]).reshape(-1, 1)
    y = np.array(car_count_values)

# Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    
    # Fit the regressor with x and y data
    regressor.fit(x, y)
    return regressor

#graph data
days = ['day'+str(d) for d in range(1,18)]
car_count_values = [101,60,55,50,52,56,79,110,56,47,54,59,60,59,87,114,0]

#database
# cred = credentials.Certificate('./ckey.json')
# firebase_admin.initialize_app(cred,{'databaseURL':'https://trafic-management-default-rtdb.firebaseio.com/'})

# state = db.reference('state')

# ref = db.reference('car_count_road1')
# get details about the video
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(0, 250), (1080, 250), (1080, 285), (0, 285)]  # line or region points

mask = cv2.imread('./mask_images/mask2.png')
prev_carCount = 0
# state.set('1')

classes_to_count = [2] 
# installing and set args for objectCounter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=model.names,
                 )

#for graph
car_count_model = train_model(days,car_count_values)
predicted_value = car_count_model.predict(np.array([17]).reshape(-1, 1))
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(days,car_count_values,marker="o",color="blue",mfc='none',label="car count",mec='black')
plt.title(f"Car count per day(predicted value of day17:{int(predicted_value)})")
plt.xlabel("days")
plt.ylabel("car count")
plt.legend([line1],["car count"])

# main functionality
while cap.isOpened():

    success, frame = cap.read()
    car_count = 0
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    img_region = cv2.bitwise_and(frame,mask)
    
    # use mask for certain region count
    tracks = model.track(img_region, show=False,
                         classes=classes_to_count,verbose=False)

    frame = counter.start_counting(frame, tracks)
    
    # count the number of prediction on the frame
    for t in tracks:
        car_count += len(t)
    
    # if prev_carCount != car_count:
        
    #     if car_count > 6:
    #         state.set('1')
    #     else:
    #         state.set('2')
            
    line1.set_xdata(days)
    car_count_values[-1]= counter.out_counts + counter.in_counts + 50
    line1.set_ydata(car_count_values)
    fig.canvas.draw()
    fig.canvas.flush_events()
    print(f"car count: {car_count}")
    print(f"total_count: {counter.in_counts + counter.out_counts}")
    prev_carCount = car_count
cap.release()
cv2.destroyAllWindows()

def algorithm(road):
    temp = road
    road_queue = []
    while road:
        road_queue.append(max(road))
        temp.remove(max(road))
        
    
    
