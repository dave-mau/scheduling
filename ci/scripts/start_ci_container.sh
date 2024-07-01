# Cleanup docker ci container, in case sth went wrong the last time.
docker stop ci-container > /dev/null 2>&1
# Create the ci-container from the scheduling image.
docker run --rm -dit --name ci-container --mount type=bind,source="$(pwd)",target=/home/ci/workspace davemau/scheduling