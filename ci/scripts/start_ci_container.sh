docker stop ci-container
docker run --rm -dit --name ci-container --mount type=bind,source="$(pwd)",target=/home/ci/workspace davemau/scheduling