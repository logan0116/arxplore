# # qdrant
# docker run \
#   -d \
#   --restart=always \
#   --name qdrant_server \
#   -p 6333:6333 \
#   -v ${PWD}/data/qdrant:/qdrant/storage \
#   qdrant/qdrant:v1.17


docker run \
  -it \
  --rm \
  --name arxplore \
  --network=host \
  --shm-size 32G \
  -v ${PWD}/arxplore:/arxplore \
  -w /arxplore \
  iss/baseenv:v1.12
  # python3 main.py