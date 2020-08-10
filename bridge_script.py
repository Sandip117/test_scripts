import os
import subprocess as sp

input_path = "s2l_out"
output_path = "bridge_out"
training_pods = ""#"label-10"

if training_pods == "":
    pods = os.listdir(input_path)
else:
    pods = training_pods.split(',')
    
for pod in pods:
    ## create a subprocess
    input_dir = "%s/%s" %(input_path,pod)
    output_dir = "%s/%s" %(output_path,pod)
    
    # Start Training
    print ("\n\n\nTraining started on %s ...." %pod)
    docker_cmd = "--rm -v %s:/incoming -v %s:/outgoing fnndsc/pl-mricnn mricnn.py --mode 1 --epochs 5 /incoming /outgoing" %(input_dir,output_dir)
    logs = sp.run(["docker", "run", "--rm", "fnndsc/pl-mricnn","mricnn.py","--mode","1", input_dir,output_dir], stdout=sp.PIPE)
    msg = str(logs.stdout).split('\n')

    print ("\n%s\n"%msg)
    print ("Training completed on %s ...." %pod)
    
