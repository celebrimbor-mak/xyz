import cv2
from consumer import Consumer
import time
import multiprocessing

if __name__ == '__main__':
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    
    # Start consumers
    num_consumers = 4
    print('Creating %d consumers' % num_consumers)
    consumers = [ Consumer(tasks)
                  for i in range(num_consumers) ]
    for w in consumers:
        w.start()

    start = int(round(time.time() * 1000))
    total = 0
    cap = cv2.VideoCapture('../song.mp4', 0,)
    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(cap.get(cv2.CAP_PROP_FPS))
    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            break
        if total%3!=0:
            total+=1
            continue 
        tasks.put(frame)
        total+=1
    
    print('Enqueued')
    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    print('Waiting')
    tasks.join()
    end = int(round(time.time() * 1000))
    print('Time taken %s' % (end-start))
    print('Completed')
