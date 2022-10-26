To train model:
1. download train data:
!curl -OL https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!curl -OL https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
!tar -xf images.tar.gz
!tar -xf annotations.tar.gz
2. run "python3 train.py" to train model and save weights

To run service:
run "python3 server.py"
