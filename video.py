# Program To Read video
# and Extract Frames
import cv2
import os
import skvideo.io

# Function to extract frames
def FrameCapture(path):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('/home/aswin/Downloads/videoplayback')
    # for frame in videogen:
    #     print(frame.shape)

    if (cap.isOpened()== False):
      print("Error opening video stream or file")

    # Read until video is completed
    ret=True
    count=0
    while(ret):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:

    # Display the resulting frame
        cv2.imwrite(os.path.join(path , "frame%d.jpg" % count), frame)
        count+=1

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

      # Break the loop
      else:
        break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

# Driver Code
if __name__ == '__main__':

    # Calling the function
    FrameCapture("/home/aswin/proj/frames")
