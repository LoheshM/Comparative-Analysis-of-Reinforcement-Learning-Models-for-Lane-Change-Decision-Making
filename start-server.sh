#! /bin/sh
ls
module load singularity/3.5.2
SINGULARITYENV_SDL_VIDEODRIVER=offscreen singularity exec --nv -e ~/carla_0.9.10.1.sif /home/carla/CarlaUE4.sh -opengl -carla-port=1555