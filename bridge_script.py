import os
import subprocess as sp
# we import the Twilio client from the dependency we just installed
from twilio.rest import Client
# the following line needs your Twilio Account SID and Auth Token
client = Client("ACd96b47674a88fede12be48c62e175471", "1667ff774f4340edb57819de19d725e5")
input_path = "s2l_out_2"
output_path = "all_trained_labels"
training_pods = ""#"label-10"

if training_pods == "":
    pods = os.listdir(input_path)
else:
    pods = training_pods.split(',')
    
for pod in pods:
    ## create a subprocess
    input_dir = "/%s/%s/" %(input_path,pod)
    output_dir = "/%s/%s" %(output_path,pod)
    
    # Start Training
    print ("\n\n\nTraining started on %s ...." %pod)
    docker_cmd = "--rm -v %s:/incoming -v %s:/outgoing fnndsc/pl-mricnn mricnn.py --mode 1 --epochs 5 /incoming /outgoing" %(input_dir,output_dir)
    try:
       logs = sp.run(["docker", "run","--gpus","1", "--rm","-v","/home/sandip/Demo/%s:/incoming" %input_dir,"-v","/home/sandip/Demo/%s:/outgoing" %output_dir, "fnndsc/pl-mricnn","mricnn.py","--mode","1","--epochs","20", "/incoming","/outgoing"], stdout=sp.PIPE)
    except:
        # change the "from_" number to your Twilio number and the "to" number
# to the phone number you signed up for Twilio with, or upgrade your
# account to send SMS to any phone number
        client.messages.create(to="6173097630",
                       from_="+12056515484",
                       body="Training failed")
    msg = str(logs.stdout).split('\n')

    print ("\n%s\n"%msg)
    print ("Training completed on %s ...." %pod)
# change the "from_" number to your Twilio number and the "to" number
# to the phone number you signed up for Twilio with, or upgrade your
# account to send SMS to any phone number
client.messages.create(to="6173097630",
                       from_="+12056515484",
                       body="Training completed/stopped")

    
