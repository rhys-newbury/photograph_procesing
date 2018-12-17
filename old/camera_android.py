

import android
import time
droid = android.Android()

delay = droid.dialogGetInput('Input 1','Delay before starting?','60').result
numberOfShots = droid.dialogGetInput('Input 2','Total images to capture?', '20').result
delayBetweenShots = droid.dialogGetInput('Input 3','Delay (Seconds) between captures','30').result
droid.ttsSpeak('taking pictures in'+  delay +'seconds')
time.sleep(int(delay))
counter = 1
droid.ttsSpeak('taking pictures now')
while counter <=int(numberOfShots):
    droid.cameraCapturePicture('/sdcard/DCIM/Camera/'+str(counter)+ ".jpg")
    counter +=1
    if counter != int(numberOfShots):
        time.sleep(int(delayBetweenShots))

print "done without error..."
droid.ttsSpeak('Finished without error...')
del droid