#!/bin/bash 
# Ensure that you have installed docker(API >= 1.40) and the nvidia graphics driver on host!

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
IMGNAME="openaigym"
CONTNAME="openaigym"
#DOCKERFILEPATH="./docker"
REPONAME="RL289A-WQ2020"
JUPYTERPORT="9000"
TENSORBOARDPORT="6006"
cd "$SCRIPTPATH"
#cd "$DOCKERFILEPATH"

USAGE="Usage: ./setup.sh [rmimcont=[0,1]] [rmimg=[0,1]]\n"
USAGE+="\trmimcont=[0,1] : 0 to not remove intermediate Docker containers\n"
USAGE+="\t                 after a successful build and 1 otherwise\n"
USAGE+="\t                 default is 1\n"
USAGE+="\trmimg=[0,1]    : 0 to not remove previously built Docker image\n"
USAGE+="\t                 and 1 otherwise\n"
USAGE+="\t                 default is 0\n"

REMOVEIMDDOCKERCONTAINERCMD="--rm=true"
REMOVEPREVDOCKERIMAGE=false

COMMANDTORUN="cd /$REPONAME && xvfb-run -s \"-screen 0 1280x720x24\" jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=$JUPYTERPORT &"
#COMMANDTOINSTALL="cd /$REPONAME/gym-sokoban && pip3 install -e ."
COMMANDTOSTARTCONTAINER="docker start -ai $CONTNAME"

# Instead, add user to group 'docker'
check_root() {
  if [ "$EUID" -ne 0 ] ; then
    echo -e "Please run this script under sudo with root permission."
    exit 1
  fi
}

test_retval() {
	if [ $? -ne 0 ] ; then
		echo -e "\nFailed to ${*}... Exiting...\n"
		exit 1
	fi
}

parse_argument() {
  # Parsing argument
  if [ $# -ne 0 ] ; then
  	while [ ! -z $1 ] ; do
  		if [ "$1" = "rmimcont=0" ] ; then
  			REMOVEIMDDOCKERCONTAINERCMD="--rm=false"
  		elif [ "$1" = "rmimg=1" ] ; then
  			REMOVEPREVDOCKERIMAGE=true
  		elif [[ "$1" != "rmimcont=1" && "$1" != "rmimg=0" ]] ; then
  			echo -e "Unknown argument: " $1
  			echo -e "$USAGE"
  			exit 1
  		fi
  		shift
  	done
  fi
}

print_setup_info() {
  # Echo the set up information
  echo -e "\n\n"
  echo -e "################################################################################\n"
  echo -e "\tSet Up Information\n"
  if [ "$REMOVEIMDDOCKERCONTAINERCMD" = "--rm=true" ] ; then
  	echo -e "\t\tRemove intermediate Docker containers after a successful build\n"
  else
  	echo -e "\t\tKeep intermediate Docker containers after a successful build\n"
  fi
  if [ "$REMOVEPREVDOCKERIMAGE" = true ] ; then
  	echo -e "\t\tCautious!! Remove previously built Docker image\n"
  else
  	echo -e "\t\tKeep previously built Docker image\n"
  fi
  echo -e "################################################################################\n"
}

remove_prev_docker_image () {
  # Remove previously built Docker image
  if [ "$REMOVEPREVDOCKERIMAGE" = true ] ; then
  	echo -e "\nRemoving previously built image..."
  	docker rmi -f $IMGNAME
  fi
}

write_command_to_enter_repo_file() {
  # Echo command to run the application
  echo "# Cyan color" > bashrc \
    && echo echo -e \"\\e[1\;36m\" >> bashrc
  echo -n COMMANDTORUN=\" >> bashrc \
    && echo -n ${COMMANDTORUN} | sed 's/\"/\\"/g' >> bashrc \
    && echo \" >> bashrc
  # Echo command to install gym_sokoban
  #echo -n COMMANDTOINSTALL=\" >> bashrc \
  #  && echo -n ${COMMANDTOINSTALL} | sed 's/\"/\\"/g' >> bashrc \
  #  && echo \" >> bashrc
  #echo echo -e \"\\n\\n\" >> bashrc
  echo echo -e \"################################################################################\\n\" >> bashrc
  #echo echo -e \"\\tCommand to install gym_sokoban for the first time:\\n\\t\\t'${COMMANDTOINSTALL}'\\n\" >> bashrc
  echo echo -e \"\\tCommand to enter repository:\\n\\t\\t'${COMMANDTORUN}'\\n\" >> bashrc
  echo echo -e \"################################################################################\\n\" >> bashrc
  echo "# Turn off colors" >> bashrc \
    && echo echo -e \"\\e[m\" >> bashrc
}

build_docker_image() {
  # Build and run the image
  echo -e "\nBuilding image $IMGNAME..."
  docker build $REMOVEIMDDOCKERCONTAINERCMD -t $IMGNAME .
  test_retval "build Docker image $IMGNAME"
  rm -rf bashrc
}

build_docker_container() {
  # Build a container from the image
  echo -e "\nRemoving older container $CONTNAME..."
  if [ 1 -eq $(docker container ls -a | grep "$CONTNAME$" | wc -l) ] ; then
  	docker rm -f $CONTNAME
  fi

  echo -e "\nBuilding a container $CONTNAME from the image $IMGNAME..."
  docker create -it --name=$CONTNAME \
    -u $(id -u):$(id -g) \
  	-v "$SCRIPTPATH":/$REPONAME \
  	-v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /etc/localtime:/etc/localtime:ro \
  	-e DISPLAY=$DISPLAY \
  	--ipc=host \
    --gpus all \
  	-p $JUPYTERPORT:$JUPYTERPORT \
  	-p $TENSORBOARDPORT:$TENSORBOARDPORT \
    --privileged=true \
  	$IMGNAME /bin/bash
  test_retval "create Docker container"
}

print_command_to_enter_repo() {
  # Echo command to run the application
  echo -e "\n\n"
  echo -e "################################################################################\n"
  echo -e "\tCommand to enter repository:\n\t\t${COMMANDTORUN}\n"
  echo -e "################################################################################\n"
}

start_docker_container() {
  docker start -ai $CONTNAME
  
  if [ 0 -eq $(docker container ls -a | grep "$CONTNAME$" | wc -l) ] ; then
  	echo -e "\nFailed to start/attach Docker container... Exiting...\n"
  	exit 1
  fi
}

print_command_to_restart_container() {
  # Echo command to start container
  echo -e "\n\n"
  echo -e "################################################################################\n"
  echo -e "\tCommand to start Docker container:\n\t\t${COMMANDTOSTARTCONTAINER}\n"
  echo -e "################################################################################\n"
}

# Check root permission
#check_root
# Parse shell script's input arguments
parse_argument "$@"
# Print the setup info
print_setup_info
# Print usage of the script
echo -e "\n$USAGE\n"

echo -e ".......... Set up will start in 5 seconds .........."
sleep 5

remove_prev_docker_image
write_command_to_enter_repo_file
build_docker_image
build_docker_container
#print_command_to_enter_repo
start_docker_container

# When exit from docker container
print_command_to_restart_container
