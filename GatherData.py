import cv2
import numpy as np
import EasyPySpin
import time
import adafruit_gps
import serial

def get_GPS(gps, last_print):
    gps.update()

    if not gps.has_fix:
        # Try again if we don't have a fix yet.
        print("Waiting for fix...")

    if gps.has_fix:
        print("Latitude: {0:.6f} degrees".format(gps.latitude))
        print("Longitude: {0:.6f} degrees".format(gps.longitude))
        print("Fix quality: {}".format(gps.fix_quality))
        # Some attributes beyond latitude, longitude and timestamp are optional
        # and might not be present.  Check if they're None before trying to use!
        if gps.satellites is not None:
            print("# satellites: {}".format(gps.satellites))
        if gps.altitude_m is not None:
            print("Altitude: {} meters".format(gps.altitude_m))
        
    return gps.latitude, gps.longitude

def start_GPS():
    uart = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=3000)
    # Create a GPS module instance.
    gps = adafruit_gps.GPS(uart)  # Use UART/pyserial
    # Turn on the basic GGA and RMC info (what you typically want)
    gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
    # Set update rate to once a second (1hz) which is what you typically want.
    gps.send_command(b"PMTK220,1000")

    return gps

def main():
    PATH = '/home/oss-carleton02/ContextualObjectRecognition_Capstone/GatheringData'
    cap = EasyPySpin.VideoCapture(0)

    if not cap.isOpened():
        print("Camera can't open\nexit")
        return -1

    cap.set(cv2.CAP_PROP_EXPOSURE, -1)  # -1 sets exposure_time to auto
    cap.set(cv2.CAP_PROP_GAIN, -1)  # -1 sets gain to auto
    gps = start_GPS()
    last_print = time.monotonic()
    i = 0
    while cap.isOpened():
        t1 = time.time()
        ret, frame = cap.read()
        frame_bgr = cv2.demosaicing(frame, cv2.COLOR_BayerBG2BGR)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)  # for RGB camera demosaicing

        lat, long = get_GPS(gps, last_print)
        #print(lat, long)
        #img_show = cv2.resize(frame_bgr, None, fx=0.75, fy=0.75)
        
        saved = cv2.imwrite(PATH +'/frame_'+str(i)+'.jpg', frame_bgr)
        if saved:
            print('image saved')
        else:
            print('image not saved')

        #cv2.imshow("press q to quit", img_show)
        #key = cv2.waitKey(30)
        #if key == ord("q"):
            #break

        i = i + 1


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()








