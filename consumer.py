from mtcnn.mtcnn import MTCNN
import multiprocessing

class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue

    def run(self):
        detector = MTCNN()
        proc_name = self.name
        print('Name %s' % proc_name)
        count=0
        while True:
            print('{}: Processed {}'.format(proc_name,count))
            frame = self.task_queue.get()
            if frame is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break
            faces = detector.detect_faces(frame)
            self.task_queue.task_done()
            count+=1
