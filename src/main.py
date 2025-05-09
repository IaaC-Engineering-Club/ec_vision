from vision import visionNav as vn
from vision import CardinalBuoys as cb
import cv2 as cv

# files

red_green = cv.VideoCapture('/home/salvador_cb/3_term/engineering_club/data/Videos/Bouygs in the sea.mp4')
green = cv.VideoCapture('/home/salvador_cb/3_term/engineering_club/data/Videos/video with only green bouyg.mp4')
red = cv.VideoCapture('/home/salvador_cb/3_term/engineering_club/data/Videos/video with only red bouyg.mp4')
empty = cv.VideoCapture('/home/salvador_cb/3_term/engineering_club/data/Videos/empty sea.mp4')

simulation_buoys = cv.VideoCapture('/home/salvador_cb/3_term/engineering_club/data/Videos/path 2.mp4')

simulation_cardinals = cv.VideoCapture('/home/salvador_cb/3_term/engineering_club/data/Videos/cradinal_buoys_path ‐ Hecho con Clipchamp.mp4')

output = "/home/salvador_cb/3_term/engineering_club/data/output/red_output_exp.mp4"

# cardinal images
north = cv.imread('/home/salvador_cb/3_term/engineering_club/data/img/cardinals/north_buoy.png')
south = cv.imread('/home/salvador_cb/3_term/engineering_club/data/img/cardinals/south_buoy.png')
east = cv.imread('/home/salvador_cb/3_term/engineering_club/data/img/cardinals/crop_east.png')
west = cv.imread('/home/salvador_cb/3_term/engineering_club/data/img/cardinals/crop_west.png')

cardinals = [east, west]
#MAIN

if __name__ == "__main__":
    
    '''nav = vn(video=simulation)
    nav.run_on_video(output_path=output)'''

    vision = cb(cardinals,video=simulation_cardinals)
    vision.run_on_video(output)