from utils import print_time

class DecectionHandler():

    def new_detection(self):
        print_time()
        print_time("-----------------------", 'green')
        print_time("-- Detection Started --", 'green')
        print_time("-----------------------", 'green')

    def end_detection(self):
        orange = (255, 192, 128)
        print_time("--- Detection Ended ---", orange)
        print_time("-----------------------", orange)
